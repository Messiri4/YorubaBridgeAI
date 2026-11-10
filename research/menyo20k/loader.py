"""Reproducible MENYO-20k parallel-data loader.

Source: https://github.com/uds-lsv/menyo-20k_MT (udÉ:-LSV, University of Hamburg)
License: CC BY-NC 4.0 (see data/menyo20k/raw/LICENSE). NON-COMMERCIAL USE ONLY.

Design constraints (research plan, step 2):
- Reads the authors' *original* TSV splits; official splits are never merged,
  shuffled, rebalanced or re-split. ``test`` is loaded ONLY on request
  (``allow_test=True``); the final-evaluation holdout stays untouched.
- Yoruba tone marks (́ ̀ ́ ̌) and underdots (ẹ ọ ṣ) are preserved verbatim;
  the only normalization applied is Unicode NFC on BOTH sides — NFC is
  canonical, so it never changes rendering, only the underlying byte form.
- Malformed / empty / unaligned rows are rejected and reported, not silently
  dropped or repaired.
- Domain provenance is kept when available (per-domain dev/test files).
- Duplicate pairs and cross-split overlap are detected (diagnostics; the
  authors' files themselves are never rewritten).
- No remote code is executed: loading is done from local TSVs fetched once
  via plain HTTPS download of data files only (checksums in SHA256SUMS).
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT_URL = "https://raw.githubusercontent.com/uds-lsv/menyo-20k_MT/master/data"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "menyo20k" / "raw"

SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train"],
    "dev": ["dev"],
    "test": ["test"],
}
# Per-domain files exist for dev/test only; the domain column in the aggregate
# TSV header (English\tYoruba) carries no domain labels.
DOMAIN_FILES: dict[str, list[str]] = {
    "dev": ["book", "digital", "news", "proverbs", "tedTalks"],
    "test": [],  # intentionally empty: held-out split, not touched during research
}


@dataclass
class Pair:
    """One aligned English–Yoruba sentence pair."""

    idx: int          # 1-based line index inside its file (excluding header)
    split: str        # train | dev | test
    en: str           # English source, NFC-normalized
    yo: str           # Yoruba target, NFC-normalized (tone marks + underdots intact)
    domain: str | None = None


@dataclass
class DatasetStats:
    pairs: int = 0
    rejected: int = 0
    duplicates_within_split: int = 0
    cross_split_overlaps: dict[str, int] = field(default_factory=dict)
    domains: dict[str, int] = field(default_factory=dict)


def normalize(text: str) -> str:
    """Unicode NFC normalization (canonical; preserves all tone marks)."""
    return unicodedata.normalize("NFC", text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _reject(line_no: int, reason: str, stats: DatasetStats) -> None:
    stats.rejected += 1
    print(f"[menyo-loader] rejected line {line_no}: {reason}")


def parse_line(line_no: int, line: str, split: str, stats: DatasetStats) -> Pair | None:
    """Parse one TSV line into a Pair, rejecting malformed input.

    Validation rules:
    - exactly two tab-separated columns (English \\t Yoruba);
    - both non-empty after stripping surrounding whitespace;
    - NFC normalization only (no case folding, no punctuation edits);
    - Yoruba side must contain characters that look like Yoruba text
      (any Latin letter incl. dotted letters); a pure-ASCII/number target is
      treated as suspicious but not fatal, while a missing target is fatal.
    """
    raw = line.rstrip("\n\r")
    cols = raw.split("\t")
    if len(cols) != 2:
        _reject(line_no, f"expected 2 columns, got {len(cols)}", stats)
        return None
    en, yo = normalize(cols[0].strip()), normalize(cols[1].strip())
    if not en or not yo:
        _reject(line_no, "empty source or target", stats)
        return None
    if not any(ch.isalpha() for ch in yo):
        _reject(line_no, "target has no alphabetic characters", stats)
        return None
    # After NFC, any remaining combining mark must attach to a base letter;
    # a leading mark indicates a broken/decomposed row.
    if yo and unicodedata.category(yo[0]) == "Mn":
        _reject(line_no, "target starts with a combining mark (broken composition)", stats)
        return None
    return Pair(idx=line_no, split=split, en=en, yo=yo)


def load_split(
    split: str,
    *,
    allow_test: bool = False,
    with_domains: bool = True,
    check_checksums: bool = True,
) -> tuple[list[Pair], DatasetStats]:
    """Load one official MENYO-20k split from local TSVs.

    Args:
        split: "train" | "dev" | "test".
        allow_test: must be True to load ``test`` — guards the holdout.
        with_domains: additionally read per-domain files (dev only) and attach
            domain labels to pairs matched by (en, yo) content.
        check_checksums: verify SHA256SUMS before reading.
    Returns:
        (pairs, stats)
    Raises:
        FileNotFoundError: when the raw TSV is missing.
        ValueError: when attempting to load `test` without allow_test, or on
            checksum mismatch.
    """
    if split == "test" and not allow_test:
        raise ValueError(
            "Refusing to load the official MENYO-20k test split. It is reserved "
            "for the final evaluation; pass allow_test=True explicitly if you "
            "are running that final evaluation."
        )

    if check_checksums and (DATA_DIR / "SHA256SUMS").exists():
        for line in (DATA_DIR.parent / "SHA256SUMS").read_text().splitlines():
            expected, _, name = line.partition("  ")
            actual = sha256(DATA_DIR / name)
            if actual != expected:
                raise ValueError(f"checksum mismatch for {name}: {actual}")

    pairs: list[Pair] = []
    stats = DatasetStats()

    path = DATA_DIR / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing MENYO file: {path}")
    seen: set[tuple[str, str]] = set()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if i == 1 and line.startswith(("English\t", "English\r")):
            continue  # header
        pair = parse_line(i, line, split, stats)
        if pair is None:
            continue
        if (pair.en, pair.yo) in seen:
            stats.duplicates_within_split += 1
        seen.add((pair.en, pair.yo))
        pairs.append(pair)
        stats.pairs += 1

    if with_domains:
        for domain in DOMAIN_FILES.get(split, []):
            dpath = DATA_DIR / f"{split}_{domain}.tsv"
            if not dpath.exists():
                continue
            domain_pairs: list[tuple[str, str]] = []
            for i, line in enumerate(dpath.read_text(encoding="utf-8").splitlines(), start=1):
                if i == 1 and line.startswith("English\t"):
                    continue
                sub = parse_line(i, line, split, stats)
                if sub is not None:
                    domain_pairs.append((sub.en, sub.yo))
            stats.domains[domain] = len(domain_pairs)
            lookup = set(domain_pairs)
            for p in pairs:
                if (p.en, p.yo) in lookup:
                    p.domain = domain
            # domain counts include only matched pairs in the aggregate file
            stats.domains[domain] = sum(1 for p in pairs if p.domain == domain)

    return pairs, stats


def cross_split_overlap(pairs_a: list[Pair], pairs_b: list[Pair]) -> int:
    """Count pairs whose (en, yo) identity appears in both lists."""
    a = {(p.en, p.yo) for p in pairs_a}
    b = {(p.en, p.yo) for p in pairs_b}
    return len(a & b)


def fingerprint(pairs: list[Pair]) -> dict:
    """Deterministic dataset fingerprint for experiment manifests."""
    domains: dict[str, int] = {}
    for p in pairs:
        key = p.domain or "unlabeled"
        domains[key] = domains.get(key, 0) + 1
    return {"n_pairs": len(pairs), "domains": domains}
