"""Reversible numeric/identifier protection for NLLB translation (Sprint TI-1).

Mechanism
---------
BEFORE generation: numeric expressions, times, dates, URLs, emails, and
identifiers are replaced with opaque sentinel placeholders (@@0@@, @@1@@, …)
that NLLB-200 copies through reliably (@@i@@ survives generation verbatim
in both directions; guillemets «i» did not).

AFTER generation: sentinels are restored to the original expressions.
Every restoration is verified:

  - placeholder count matches exactly, and each placeholder index appears
    exactly once (no missing / duplicated / altered sentinels)
  - the verification failure path is a documented SAFE FALLBACK: the
    ORIGINAL model output is returned unchanged (no silent corruption, no
    partial restoration)

Reliability hardening (numeric-protection feature branch)
--------------------------------------------------------
1. SINGLE-PASS combined alternation: all expressions are matched by ONE
   regex whose alternatives are ordered most-specific first. Because
   ``finditer`` never rescans replaced spans, nested sentinels are
   structurally impossible (Bug TI-1 root cause: earlier multi-pass
   design let later passes match inside earlier sentinels, e.g. splitting
   "-12.5" into "-<<0>>.5").
2. Sentinel-collision protection: user text that already contains literal
   @@N@@ tokens is rewritten into freshly issued sentinels whose stored
   original IS the literal token. The protected text therefore contains
   ONLY sequentially issued sentinels — a user's literal "@@3@@" can
   neither collide with a generated sentinel nor be destroyed, and user
   text that merely RESEMBLES a placeholder is never rewritten by
   restore(): only sentinels actually issued by protect() substitute.
3. Positional numbering: sentinel indexes follow left-to-right text order,
   independent of pattern priority (Bug TI-2 fix).
4. Structured failure reporting: RestorationFailure carries the failure
   kind, expected/found counts, missing and duplicated sentinel lists —
   enough for callers (and integration) to decide between surfacing an
   error and retrying without protection.
5. Reusable fallback helper: with_unprotected_fallback() runs a protected
   translation and retries it unprotected when restoration fails, without
   the translation service needing sentinel details.

Configuration
-------------
Enabled per-request via `protect_numerics=True` on
`TranslationService.translate`, or globally via settings
`numeric_protection_enabled` (default: enabled). When disabled the
pipeline is a strict pass-through.

Yoruba numerals note: the protector intentionally passes numbers as-is
(digits are valid in modern Yoruba orthography and used in FLORES and
news registers); it does NOT convert digits to Yoruba number words.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

# Sentinel format: @@<i>@@. Measured against the real model (Sprint TI-1
# diagnostics): guillemets «i» are normalized away by NLLB's SentencePiece
# (the model emitted bare digits), while @@i@@ survives generation verbatim
# in both directions. Never use characters the tokenizer merges.
SENTINEL_FMT = "@@{i}@@"
SENTINEL_RE = re.compile(r"@@\s*(\d+)@@")
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"  # 1,000.50 / 9.000,5-handled separately

# ONE combined pattern, alternatives ordered most-specific first. The
# regex engine commits to the FIRST alternative that matches at each
# scan position, so priority = order and overlapping matches (a score
# inside a date, digits inside a sentinel) cannot occur.
#
# Order rationale:
#   email/url  – contain digits; must win over every numeric rule
#   dates      – "2019-06-04" would otherwise match as score/range
#   times      – "11:29 pm"; must win over the colon-score alternative
#                (when both readings exist the time reading is the
#                safer default); bare "2:1" fails the time alternative
#                (needs two minute digits) and falls through to scores
#   scores     – "7–2", "3 - 1", "2:1"
#   currency   – "$1,000", "₦5,000", "500 naira"
#   thousands  – "1,000.50", "9.000,5", "-12.5"
#   percentage – "50%", "50 %"
#   catch-all  – signed integers/decimals ("3", "-7", "3.5")
_COMPILED: list[re.Pattern[str]] = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b(?:https?://|www\.)[^\s]+", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?:\s?(?:am|pm|AM|PM))?\b"),
    re.compile(r"(?<![\w:])(?<!\d[-–:])\d{1,4}\s?[–\-:]\s?\d{1,4}\b(?![\w:])(?![:\d])"),
    re.compile(
        r"[$€£₦¥]\s?-?\d+(?:,\d{3})*(?:\.\d+)?"
        r"|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:dollars|euros|naira|pounds|yen)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\w.])-?\d+\.\d+(?!\d)"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?"),
]
_MASTER_RE = re.compile(
    "|".join(f"(?:{p.pattern})" for p in _COMPILED),
    re.IGNORECASE,
)


@dataclass
class ProtectionResult:
    protected_text: str
    placeholders: dict[str, str] = field(default_factory=dict)  # sentinel -> original

    @property
    def n(self) -> int:
        return len(self.placeholders)


@dataclass(frozen=True)
class RestorationFailure:
    """Structured description of a failed sentinel restoration.

    kind is one of:
      "count_mismatch"        – sentinel count differs from expected
      "duplicated"            – some sentinel index appears more than once
      "missing"               – some expected sentinel index is absent
      "missing_and_duplicated"– both defects at once (most actionable)
      "unknown_sentinel"      – output contains an index protect() never issued
    """

    kind: str
    detail: str
    expected: int = 0
    found: int = 0
    missing: tuple[str, ...] = ()
    duplicated: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    def __str__(self) -> str:  # keep logs readable
        return f"{self.kind}: {self.detail}"


T = TypeVar("T")


@dataclass
class RestorationOutcome(Generic[T]):
    """Result of restoring (or attempting to restore) sentinels.

    On success: ok=True, text=restored text, failure=None.
    On failure: ok=False, text=<SAFE FALLBACK output>, failure populated.
    The fallback text is returned so callers never receive a partially
    restored string, but still have something deterministic to act on.
    """

    text: str
    ok: bool
    failure: RestorationFailure | None = None
    payload: T | None = None


def protect(source: str) -> ProtectionResult:
    """Replace supported expressions with sentinels; record originals.

    Single-pass matching means nested sentinels are impossible, and every
    match is the longest/most-specific expression at its position.
    """
    provisional: dict[str, str] = {}

    # Pass 0: rewrite user-typed @@N@@ tokens into fresh provisional
    # sentinels (original preserved verbatim for restoration). This keeps
    # the protected text free of foreign sentinel indexes.
    out: list[str] = []
    last = 0
    for m in SENTINEL_RE.finditer(source):
        out.append(source[last:m.start()])
        token = SENTINEL_FMT.format(i=len(provisional))
        provisional[token] = m.group(0)
        out.append(token)
        last = m.end()
    out.append(source[last:])
    text = "".join(out)

    def _mark(m: re.Match[str]) -> str:
        token = SENTINEL_FMT.format(i=len(provisional))
        provisional[token] = m.group(0)
        return token

    # Mask user-claimed sentinels so _MASTER_RE cannot match their digits.
    masked = re.sub(
        r"@@\s*(\d+)@@",
        lambda m: f"\x00{chr(ord('A') + int(m.group(1)))}\x01" if int(m.group(1)) < 10 else m.group(0),
        text,
    )

    protected = _MASTER_RE.sub(_mark, masked)

    # Unmask and renumber everything by text position (Bug TI-2 fix).
    combined = re.sub(r"\x00([A-J])\x01", lambda m: SENTINEL_FMT.format(i=int(ord(m.group(1)) - ord("A"))), protected)

    order = [m.group(0) for m in SENTINEL_RE.finditer(combined)]
    remap = {old: new_idx for new_idx, old in enumerate(order)}
    final_placeholders = {
        SENTINEL_FMT.format(i=remap[k]): v for k, v in provisional.items()
    }
    combined = SENTINEL_RE.sub(lambda m: SENTINEL_FMT.format(i=remap[m.group(0)]), combined)

    return ProtectionResult(protected_text=combined, placeholders=final_placeholders)


def analyze_placeholder_structure(
    output: str, expected: dict[str, str] | int
) -> RestorationFailure | None:
    """Return a structured RestorationFailure, or None if structure is sound.

    Accepts either the ProtectionResult placeholder map (enables the
    "unknown sentinel index" check) or just the expected count.
    """
    if isinstance(expected, int):
        expected_map: dict[str, str] | None = None
        expected_n = expected
    else:
        expected_map = expected
        expected_n = len(expected)

    found = SENTINEL_RE.findall(output)
    # Normalize whitespace variants the sentinel regex tolerates
    # (e.g. "@@ 3@@" emitted by the model).
    counts: dict[str, int] = {}
    for tok in found:
        key = SENTINEL_FMT.format(i=tok)
        counts[key] = counts.get(key, 0) + 1

    missing = tuple(
        SENTINEL_FMT.format(i=i)
        for i in range(expected_n)
        if SENTINEL_FMT.format(i=i) not in counts
    )
    duplicated = tuple(
        sorted(k for k, v in counts.items() if v > 1 and k not in missing)
    )

    # Precise classification first: a count mismatch is usually a SYMPTOM
    # of a specific missing/duplicated sentinel, which is actionable.
    if missing and duplicated:
        return RestorationFailure(
            kind="missing_and_duplicated",
            detail=f"missing sentinels: {list(missing)}; duplicated: {list(duplicated)}",
            expected=expected_n,
            found=len(found),
            missing=missing,
            duplicated=duplicated,
        )
    if missing:
        return RestorationFailure(
            kind="missing",
            detail=f"missing sentinels: {list(missing)}",
            expected=expected_n,
            found=len(found),
            missing=missing,
        )
    if duplicated:
        return RestorationFailure(
            kind="duplicated",
            detail=f"duplicated sentinels: {list(duplicated)}",
            expected=expected_n,
            found=len(found),
            duplicated=duplicated,
        )
    if expected_map is not None:
        unknown = tuple(sorted(set(counts) - set(expected_map)))
        if unknown:
            return RestorationFailure(
                kind="unknown_sentinel",
                detail=f"sentinel indexes not issued by protect(): {list(unknown)}",
                expected=expected_n,
                found=len(found),
                unknown=unknown,
            )
    if len(found) != expected_n:
        return RestorationFailure(
            kind="count_mismatch",
            detail=f"expected {expected_n} sentinels, found {len(found)}",
            expected=expected_n,
            found=len(found),
        )
    return None


def verify_placeholder_structure(output: str, expected_n: int) -> tuple[bool, str]:
    """Structural verification of the model output's sentinel usage.

    Backwards-compatible wrapper around analyze_placeholder_structure().
    """
    failure = analyze_placeholder_structure(output, expected_n)
    return (False, str(failure)) if failure else (True, "ok")


def _norm_key(tok: str) -> str:
    return SENTINEL_FMT.format(i=tok)


def restore_outcome(model_output: str, protection: ProtectionResult) -> RestorationOutcome:
    """restore() with a structured RestorationFailure on the failure path."""
    if protection.n == 0:
        return RestorationOutcome(text=model_output, ok=True)
    failure = analyze_placeholder_structure(model_output, protection.placeholders)
    if failure is not None:
        return RestorationOutcome(text=model_output, ok=False, failure=failure)
    restored = SENTINEL_RE.sub(
        lambda m: protection.placeholders[_norm_key(m.group(1))],
        model_output,
    )
    return RestorationOutcome(text=restored, ok=True)


def restore(model_output: str, protection: ProtectionResult) -> tuple[str, bool, str]:
    """Restore sentinels to originals.

    Returns (text, ok, reason). On ANY structural failure the ORIGINAL model
    output is returned unchanged (safe fallback: never a partial or guessed
    restoration). See restore_outcome() for the structured-failure variant.
    """
    outcome = restore_outcome(model_output, protection)
    reason = "ok" if outcome.ok else str(outcome.failure)
    return outcome.text, outcome.ok, reason


# ---------------------------------------------------------------------------
# Reusable fallback helper (integration support)
# ---------------------------------------------------------------------------

def with_unprotected_fallback(
    translate_protected: Callable[..., T],
    translate_plain: Callable[..., T],
    *args: object,
    attempt_fallback: Callable[[RestorationOutcome, str], bool] | None = None,
    on_failure: Callable[[RestorationFailure, str], None] | None = None,
    **kwargs: object,
) -> tuple[T, bool, RestorationFailure | None]:
    """Run a protected translation; retry unprotected if restoration failed.

    Designed for integration into TranslationService.translate without that
    module needing to know sentinel details:

        translated, used_fallback, failure = with_unprotected_fallback(
            lambda **kw: self._translate_once(**kw, protect_numerics=True),
            lambda **kw: self._translate_once(**kw, protect_numerics=False),
            text=..., source_language=..., target_language=...)

    `translate_protected` must return either a RestorationOutcome or an
    object exposing `.ok`, `.failure`, `.text` and optionally `.payload`.
    `translate_plain` is called ONLY when the protected attempt fails
    verification and `attempt_fallback` (default: always retry) approves.
    If `translate_plain` accepts keyword arguments it also receives
    ``protect_numerics=False``; closed-over zero-arg callables work too.

    Returns (result, used_fallback, failure).
    """
    result = translate_protected(*args, **kwargs)
    # The carried restoration outcome wins when present; the raw return
    # value is used as-is when the callable returns an outcome directly.
    payload = getattr(result, "payload", None)
    outcome = payload if isinstance(payload, RestorationOutcome) else (
        result if isinstance(result, RestorationOutcome) else payload or result
    )
    # Success returns the caller's value unchanged (payload preferred).
    failure = getattr(outcome, "failure", None)
    if getattr(outcome, "ok", True) and failure is None:
        return result, False, None

    text = getattr(outcome, "text", "")
    if on_failure is not None and failure is not None:
        on_failure(failure, text)

    if attempt_fallback is not None and not attempt_fallback(outcome, text):
        return result, False, failure

    logger.warning("numeric protection failed (%s); retrying without protection",
                   failure or "unknown")
    plain_kwargs = dict(kwargs)
    plain_kwargs.pop("protect_numerics", None)
    try:
        return (translate_plain(*args, **plain_kwargs, protect_numerics=False),
                True, failure)
    except TypeError:
        # Zero-arg (closed-over) callables cannot accept the flag.
        return translate_plain(*args, **plain_kwargs), True, failure
