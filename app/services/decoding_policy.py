"""Configurable NLLB decoding policy (Agent 4, feature/nllb-optimization).

This module centralizes beam-search / length / truncation decisions for
NLLB generation so they can be varied **as a named, validated policy**
instead of being edited ad hoc inside ``translation.py``.

Design constraints
------------------
- **Backward compatible by construction.** The default policy
    ("production_default") reproduces exactly the generation parameters the
    service already uses from settings:
    ``num_beams=settings.translation_num_beams``,
    ``max_new_tokens=settings.translation_max_new_tokens``,
    ``length_penalty=settings.translation_length_penalty`` — i.e. beams=4,
    max_new=256, lp=1.1 on stock config. Passing a policy's kwargs to
    ``model.generate`` adds nothing and changes nothing unless a policy
    profile is explicitly selected.
- **No inference, no downloads.** The module only computes argument
    dictionaries; callers hand them to ``model.generate``.
- **Conservative defaults.** Every optional feature (repetition penalty,
    dynamic output budget, early stopping tweaks) is OFF by default, so
    adopting the module cannot silently alter translations. Each knob must
    be turned on deliberately, and each candidate knob has a documented
    reason to be A/B-tested against real NLLB before promotion.

Measured context (existing artifacts, n=30 FLORES dev pairs, CPU;
do NOT treat as evidence beyond that subset):

- ``data/eval/decoding_exp_dev.json``
    baseline beams=4/maxnew=256/lp=1.0  → chrF++ 26.20, 12.7 s/pair
    beams=4/maxnew=400/lp=1.1           → chrF++ 26.45, 15.1 s/pair
    beams=8/maxnew=400/lp=1.1           → chrF++ 25.87, 20.6 s/pair
- ``data/eval/phase3_scorecard_sprint2a.csv``: beams=4 beats beams=2 for
  eng→yor (26.75 vs 26.07 chrF++) but *loses* slightly for yor→eng
  (43.01 vs 43.56) while costing ~30% more latency — motivates a
  **direction-aware** policy knob, default OFF.
- ``docs/translation_evaluation.md``: 0/30 dev hypotheses truncated at
  max_new_tokens=256, so output length limits are adequate for
  FLORES-length sentences; a larger budget is only justified for long
  inputs (guard against mid-sentence truncation, not a free win).

Public surface
--------------
- ``DecodingPolicy`` — frozen, validated dataclass.
- ``PROFILES`` — named preset policies (all validated at import).
- ``get_policy(settings=None, profile=None)`` — resolution order:
  explicit profile name > settings override > "production_default".
- ``build_generate_kwargs(policy, settings, input_tokens)`` — the single
  function a caller passes its ``model.generate(**kwargs)`` args through.
- ``EXPERIMENT_MATRIX`` / ``experiment_plan_md`` — defined-but-not-run A/B
  configurations for a future real-NLLB session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodingPolicy:
    """Validated description of one NLLB decoding configuration.

    Attributes:
        name: Stable identifier used in experiment logs and reports.
        num_beams: Beam width (1 = greedy). Range [1, 8]; 8+ was measured
            worse *and* slower on the dev subset, so validation caps it.
        max_new_tokens: Hard cap on generated tokens (output-length limit
            guarding against runaway/hallucinated continuations).
        length_penalty: HF beam-search length normalization exponent.
            >1.0 favors longer hypotheses; the adopted production value
            is 1.1 (small eng→yor chrF++ gain, no truncation cost).
        repetition_penalty: Multiplicative penalty on already-emitted
            tokens (1.0 = disabled). WARNING: Yoruba uses legitimate
            reduplication ("ọjọ ọjọ" = day after day), so a penalizer can
            damage grammatical outputs; default stays 1.0 and any value
            must clear the existing repetition detector, not a human eye.
        no_repeat_ngram_size: Block repeated n-grams (0 = disabled).
            Same reduplication caveat; conservative default 0.
        early_stopping: HF beam-search ``early_stopping`` flag; None means
            "let the transformers default decide" (current behavior).
        min_new_tokens: Floor on output length to discourage premature
            short outputs (0 = disabled; matches current behavior).
        dynamic_output_budget: When True, ``build_generate_kwargs`` scales
            ``max_new_tokens`` from the actual input length instead of
            using the flat cap. OFF by default — flat 256 is measured
            sufficient for FLORES-length sentences, and the interaction
            with truncation metrics must be re-measured before enabling.
        budget_input_ratio / budget_min / budget_max: Parameters of the
            dynamic budget ``clamp(ratio * input_tokens, min, max)``.
        direction_aware_beams: Map of ``"src_tgt"`` → beam width overrides.
            Motivated by the beams=2-vs-4 split finding above, but OFF
            (empty map) until confirmed on larger samples.
        notes: Free-text provenance (which evidence motivated the value).
    """

    @property
    def experimental_active(self) -> bool:
        """True when any non-production knob is engaged.

        Used by ``build_generate_kwargs``: only a fully-default policy
        defers to ambient settings; experiments are authoritative.
        """
        return (
            self.num_beams != 4
            or self.max_new_tokens != 256
            or self.length_penalty != 1.1
            or self.dynamic_output_budget
            or bool(self.direction_aware_beams)
        )

    name: str
    num_beams: int = 4
    max_new_tokens: int = 256
    length_penalty: float = 1.1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool | None = None
    min_new_tokens: int = 0
    dynamic_output_budget: bool = False
    budget_input_ratio: float = 2.0
    budget_min: int = 64
    budget_max: int = 512
    direction_aware_beams: Mapping[str, int] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not (1 <= self.num_beams <= 8):
            errors.append(f"num_beams={self.num_beams} outside [1, 8]")
        if not (16 <= self.max_new_tokens <= 1024):
            errors.append(f"max_new_tokens={self.max_new_tokens} outside [16, 1024]")
        if not (0.0 <= self.length_penalty <= 2.0):
            errors.append(f"length_penalty={self.length_penalty} outside [0.0, 2.0]")
        if not (1.0 <= self.repetition_penalty <= 2.0):
            errors.append(f"repetition_penalty={self.repetition_penalty} outside [1.0, 2.0]")
        if not (0 <= self.no_repeat_ngram_size <= 10):
            errors.append(f"no_repeat_ngram_size={self.no_repeat_ngram_size} outside [0, 10]")
        if self.min_new_tokens < 0 or self.min_new_tokens >= self.max_new_tokens:
            errors.append(f"min_new_tokens={self.min_new_tokens} must be in [0, max_new_tokens)")
        if self.dynamic_output_budget and not (
            self.budget_min <= self.budget_max <= self.max_new_tokens * 4
        ):
            errors.append("require budget_min <= budget_max <= 4*max_new_tokens")
        if self.budget_input_ratio <= 0:
            errors.append("budget_input_ratio must be positive")
        for key, beams in self.direction_aware_beams.items():
            # Key format: '<src>_<tgt>' using ISO-639-3 script codes as used
            # elsewhere in the repo, e.g. 'eng_yor', 'yor_eng'.
            parts = key.split("_")
            if (
                key.count("_") != 1
                or not all(p.isalpha() and len(p) >= 3 for p in parts)
            ):
                errors.append(
                    f"direction_aware_beams key '{key}' must be '<src>_<tgt>' "
                    "language codes (e.g. 'eng_yor')"
                )
            elif not (1 <= beams <= 8):
                errors.append(f"direction_aware_beams['{key}']={beams} outside [1, 8]")
        if errors:
            raise ValueError(f"invalid decoding policy '{self.name}': " + "; ".join(errors))

    # -- derived behaviour ---------------------------------------------------

    def effective_max_new_tokens(self, input_tokens: int | None = None) -> int:
        """Output-token budget for a given input length.

        With the dynamic budget disabled (default) this is exactly the flat
        cap, matching current generation behavior. Enabled, it scales with
        input length and clamps, so long inputs get headroom to finish
        instead of being cut mid-sentence while short inputs stay fast.
        """
        if not self.dynamic_output_budget or input_tokens is None:
            return self.max_new_tokens
        raw = round(self.budget_input_ratio * input_tokens)
        return max(self.budget_min, min(self.budget_max, raw))

    def effective_num_beams(self, direction: str | None = None) -> int:
        """Beam width, honoring optional per-direction overrides."""
        if direction and self.direction_aware_beams:
            return self.direction_aware_beams.get(direction, self.num_beams)
        return self.num_beams

    def with_overrides(self, **changes: Any) -> "DecodingPolicy":
        """Derive a named variant (used by the experiment matrix)."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Resolution & generate-kwargs bridge
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = "production_default"


def get_policy(
    settings: Any | None = None,
    profile: str | None = None,
    env_var: str = "DECODING_POLICY",
) -> DecodingPolicy:
    """Resolve a policy: explicit arg > settings/env override > default.

    ``settings`` may carry ``decoding_policy_profile`` (a string) so the
    service can opt in via config without this module importing
    ``app.core.config``. Unknown names raise ValueError (fail loud, never
    silently fall back — silent fallbacks hide experiment contamination).
    """
    chosen = profile
    if chosen is None and settings is not None:
        chosen = getattr(settings, "decoding_policy_profile", None)
    if chosen is None:
        chosen = os.environ.get(env_var)
    if chosen is None:
        chosen = DEFAULT_PROFILE
    if chosen not in PROFILES:
        raise ValueError(
            f"unknown decoding policy '{chosen}'. Available: {sorted(PROFILES)}"
        )
    return PROFILES[chosen]


def build_generate_kwargs(
    policy: DecodingPolicy,
    settings: Any | None = None,
    input_tokens: int | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    """Build the exact keyword arguments for ``model.generate``.

    Backward compatibility: with the default policy and stock settings the
    dict contains precisely the four keys the service already passes
    (``max_new_tokens``, ``num_beams``, ``length_penalty`` and whatever the
    caller supplies separately, e.g. ``forced_bos_token_id``) with the same
    values. Extra generation options appear **only** when the corresponding
    policy knob is non-default, so ``model.generate`` is never surprised by
    a None.

    Settings attributes are read defensively (``getattr`` with the policy's
    own value as fallback) so the module works with mock settings and with
    older config objects lacking newer fields. Settings are honored only for
    the three production keys, and only while the policy keeps the values
    those keys had at adoption time (beams=4, max_new=256, lp=1.1) — i.e.
    while the policy is faithful to current behavior. A non-default policy
    is authoritative: it is an experiment, and it must not be silently
    reshaped by ambient settings.
    """
    max_new = policy.effective_max_new_tokens(input_tokens)
    num_beams = policy.effective_num_beams(direction)
    length_penalty = policy.length_penalty

    if settings is not None and not policy.experimental_active:
        # Production-default fidelity: mirror the settings-driven behavior,
        # including operator overrides via .env (same keys the service uses).
        max_new = int(
            getattr(settings, "translation_max_new_tokens", max_new) or max_new
        )
        num_beams = int(
            getattr(settings, "translation_num_beams", policy.num_beams)
        )
        length_penalty = float(
            getattr(settings, "translation_length_penalty", policy.length_penalty)
        )
    # translation_max_input_chars historically governed tokenizer truncation
    # only; it is owned by the tokenizer call (translation.py) and is not
    # touched here.

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
    }
    if policy.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = policy.repetition_penalty
    if policy.no_repeat_ngram_size:
        kwargs["no_repeat_ngram_size"] = policy.no_repeat_ngram_size
    if policy.early_stopping is not None:
        kwargs["early_stopping"] = policy.early_stopping
    if policy.min_new_tokens:
        kwargs["min_new_tokens"] = policy.min_new_tokens
    return kwargs


# ---------------------------------------------------------------------------
# Named profiles (all conservative unless noted)
# ---------------------------------------------------------------------------

PRODUCTION_DEFAULT = DecodingPolicy(
    name="production_default",
    num_beams=4,
    max_new_tokens=256,
    length_penalty=1.1,
    notes=(
        "Bit-for-bit equivalent to current settings-driven generation "
        "(beams=4, max_new=256, lp=1.1). All experimental knobs OFF."
    ),
)

CONSERVATIVE_GUARDS = DecodingPolicy(
    name="conservative_guards",
    num_beams=4,
    max_new_tokens=400,
    length_penalty=1.1,
    dynamic_output_budget=True,
    budget_input_ratio=2.0,
    budget_min=64,
    budget_max=400,
    notes=(
        "Same beam/length-penalty regime, but long inputs get a scaled "
        "output budget (≤400) to avoid mid-sentence truncation; short "
        "inputs keep the 256-ish behavior. Candidate only — must be shown "
        "to introduce no regressions on the 30-pair dev set first."
    ),
)

QUALITY_CANDIDATE_LP12 = DecodingPolicy(
    name="quality_lp12",
    num_beams=4,
    max_new_tokens=400,
    length_penalty=1.2,
    notes=(
        "lp=1.2 extends the measured 1.0→1.1 trend one step. beams=8 at "
        "lp=1.1 already regressed, so beam width is deliberately held at 4."
    ),
)

FAST_LATENCY = DecodingPolicy(
    name="fast_latency",
    num_beams=2,
    max_new_tokens=256,
    length_penalty=1.1,
    notes=(
        "~20% lower measured latency (25.7 vs 30.5 s mean, Sprint-2A "
        "scorecard) with slightly lower eng→yor quality; a fallback for "
        "degraded/CPU-constrained deployments, not a default."
    )
    ,
)

DIRECTION_AWARE = DecodingPolicy(
    name="direction_aware_beams",
    num_beams=4,
    direction_aware_beams={"yor_eng": 2, "eng_yor": 4},
    max_new_tokens=256,
    length_penalty=1.1,
    notes=(
        "Exploits the Sprint-2A split: beams=2 matched beams=4 for yor→eng "
        "(43.56 vs 43.01 chrF++) at ~25% less latency, while eng→yor "
        "needed beams=4. Direction-aware only; default remains uniform."
    ),
)

ANTI_REPEAT_GENTLE = DecodingPolicy(
    name="anti_repeat_gentle",
    num_beams=4,
    max_new_tokens=256,
    length_penalty=1.1,
    no_repeat_ngram_size=3,
    notes=(
        "Gentle degeneration guard. Risk: Yoruba reduplication "
        "('ọjọ ọjọ') and CV constructs; must be re-scored with the "
        "existing repetition detector whitelist before promotion."
    ),
)

PROFILES: dict[str, DecodingPolicy] = {
    p.name: p
    for p in (
        PRODUCTION_DEFAULT,
        CONSERVATIVE_GUARDS,
        QUALITY_CANDIDATE_LP12,
        FAST_LATENCY,
        DIRECTION_AWARE,
        ANTI_REPEAT_GENTLE,
    )
}

# ---------------------------------------------------------------------------
# Controlled A/B experiment plan (defined here; executed later with real NLLB)
# ---------------------------------------------------------------------------

EXPERIMENT_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "id": "A0-baseline-fidelity",
        "policy": PRODUCTION_DEFAULT.name,
        "question": "Does routing generation through decoding_policy reproduce "
        "byte-identical predictions vs the pre-module service? Gate: identical "
        "outputs on all saved FLORES dev pairs; any diff is a bug.",
        "pass_criteria": "0 divergent hypotheses vs data/eval beam-A/B predictions.",
    },
    {
        "id": "A1-dynamic-budget",
        "policy": CONSERVATIVE_GUARDS.name,
        "question": "Does an input-scaled output budget eliminate residual "
        "mid-sentence truncation on long dev/test sentences without quality loss?",
        "pass_criteria": "truncated hypotheses ↓ and chrF++ within −0.2 of A0 "
        "on both directions; latency increase ≤10%.",
    },
    {
        "id": "A2-length-penalty",
        "policy": QUALITY_CANDIDATE_LP12.name,
        "question": "Extends the measured lp 1.0→1.1 improvement one step.",
        "pass_criteria": "chrF++ ≥ A0 both directions; no truncation increase.",
    },
    {
        "id": "A3-direction-aware-beams",
        "policy": DIRECTION_AWARE.name,
        "question": "Can yor→eng keep quality at beams=2 (saving ~25% latency) "
        "while eng→yor stays at beams=4?",
        "pass_criteria": "yor→eng chrF++ ≥ A0 − 0.3 AND mean latency ↓ ≥15%; "
        "eng→yor unchanged.",
    },
    {
        "id": "A4-anti-repeat",
        "policy": ANTI_REPEAT_GENTLE.name,
        "question": "Does no_repeat_ngram_size=3 reduce genuine repetition loops "
        "without breaking attested reduplication?",
        "pass_criteria": "adjacent-repeat warnings ↓; reduplication-whitelist "
        "sentences unchanged or improved; chrF++ within −0.2 of A0.",
    },
    {
        "id": "A5-fast-fallback",
        "policy": FAST_LATENCY.name,
        "question": "Latency-oriented fallback profile for overload conditions.",
        "pass_criteria": "documented quality delta only; never promoted to default.",
    },
)


def experiment_plan_md() -> str:
    """Render the A/B matrix as a Markdown table (for the feature doc)."""
    rows = [
        f"| {e['id']} | `{e['policy']}` | {e['question']} | {e['pass_criteria']} |"
        for e in EXPERIMENT_MATRIX
    ]
    return (
        "| id | policy | question | promotion criteria |\n"
        "|---|---|---|---|\n" + "\n".join(rows)
    )


def load_policy_file(path: str | Path) -> DecodingPolicy:
    """Load a custom policy from a JSON file (for one-off experiments).

    Recognized keys mirror the ``DecodingPolicy`` fields; ``name`` is
    required so logs stay traceable.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "name" not in data:
        raise ValueError(f"policy file {path} missing required 'name'")
    known = {f for f in DecodingPolicy.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"policy file {path} has unknown keys: {sorted(unknown)}")
    return DecodingPolicy(**data)
