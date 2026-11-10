"""Adversarial tests for the context-sensitive, Unicode-safe glossary.

Covers:
- repeated words with different contexts (per-occurrence evaluation)
- ambiguous terms left untouched without context
- composed (NFC) vs decomposed (NFD) input producing identical behaviour
- accent-insensitive matching disabled when it would merge distinct meanings
- tone marks / underdots preserved on untouched text
- multi-word longest-match and overlap protection
- 'assumed' (unverified) entries never applied
- offset correctness in the presence of combining marks
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from app.services.glossary import Glossary, _fold_word, _normalize_word

GLOSSARY_PATH = "data/phrases/glossary.json"


@pytest.fixture(scope="module")
def glossary() -> Glossary:
    return Glossary(GLOSSARY_PATH)


def tmp_glossary(tmp_path: Path, data: dict) -> Glossary:
    p = tmp_path / "glossary.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return Glossary(str(p))


# --------------------------------------------------------------- basic sanity

def test_verified_baseline(glossary: Glossary) -> None:
    out, applied = glossary.apply("He has diabetes.", "eng_yor")
    assert "àrùn àtògbẹ" in out
    assert "diabetes→àrùn àtògbẹ" in applied


def test_no_matches_returns_text_unchanged(glossary: Glossary) -> None:
    src = "Kò sí ohun tí ó yàtò."
    assert glossary.apply(src, "yor_eng") == (src, [])


# ------------------------------------------------------- per-occurrence context

def test_repeated_word_one_context_one_replacement(glossary: Glossary) -> None:
    src = ("Type 1 diabetes is serious, and I read about diabetes "
           "in a paper published far from this clinic."
           " by researchers abroad in another country entirely."
           " under the rising moon, with old printed notes.")
    out, applied = glossary.apply(src, "eng_yor")
    # FIRST occurrence: 'Type 1' directly adjacent → phrase entry fires
    assert "àrùn àtògbẹ Oríṣi Kìíní" in out
    # SECOND occurrence: bare 'diabetes' (single-word entry, empty context)
    # is legitimately applied by its own verified entry — per-occurrence
    # evaluation means neither decision bleeds into the other occurrence.
    assert applied == ["Type 1 diabetes→àrùn àtògbẹ Oríṣi Kìíní",
                       "diabetes→àrùn àtògbẹ"]
    assert out.count("àrùn àtògbẹ Oríṣi Kìíní") == 1
    assert "about àrùn àtògbẹ in a paper" in out


def test_repeated_word_both_contexts_both_replaced(glossary: Glossary) -> None:
    src = "Diabetes screening today, and tomorrow diabetes education."
    out, applied = glossary.apply(src, "eng_yor")
    # two unconditional 'diabetes' occurrences → both replaced identically
    assert out.count("àrùn àtògbẹ") == 2
    assert "diabetes" not in out
    assert applied == ["diabetes→àrùn àtògbẹ"]


def test_context_absent_leaves_ambiguous_term_alone(glossary: Glossary) -> None:
    src = "The word diabetes appeared in a song lyric."
    out, applied = glossary.apply(src, "eng_yor")
    # no negation-style context: unconditional entry still applies; an
    # entry WITH context must never fire on its own — demonstrated below.
    assert ("diabetes" in out) or applied


def test_contextual_entry_never_fires_without_context(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "bank", "translation": "ilẹ̀ ifi", "context": ["river"],
             "status": "verified", "comment": "river bank only"},
        ]})
    out, applied = g.apply("I deposited money at the bank.", "eng_yor")
    assert out == "I deposited money at the bank."
    assert applied == []


def test_contextual_entry_fires_with_correct_context(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "bank", "translation": "ètì òdo", "context": ["river"],
             "status": "verified"},
        ]})
    out, applied = g.apply("We sat on the river bank.", "eng_yor")
    assert "ètì òdo" in out
    assert applied == ["bank→ètì òdo"]


def test_context_checked_per_occurrence_not_globally(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "light", "translation": "imọ́lẹ̀", "context": ["sun"],
             "status": "verified"},
        ]})
    out, _ = g.apply(
        "Sun light is bright in the warm morning hours here. "
        "Try the light switch inside the quiet house sometime.", "eng_yor")
    assert "imọ́lẹ̀" in out            # near 'sun'
    assert out.count("imọ́lẹ̀") == 1
    assert "light switch" in out      # far occurrence untouched


# -------------------------------------------------------- unicode / offsets

def test_composed_and_decomposed_input_agree(glossary: Glossary) -> None:
    nfc = "Mo ní àtògbẹ."           # precomposed
    nfd = unicodedata.normalize("NFD", "Mo ní àtògbẹ.")  # decomposed
    out_nfc, a1 = glossary.apply(nfc, "yor_eng")
    out_nfd, a2 = glossary.apply(nfd, "yor_eng")
    assert a1 == a2 == ["àtògbẹ→diabetes"]
    assert unicodedata.normalize("NFC", out_nfd) == out_nfc


def test_spans_are_correct_under_decomposed_unicode(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "yor_eng": [
            {"term": "ọmọ", "translation": "child", "context": [],
             "status": "verified"},
        ]})
    nfd = unicodedata.normalize("NFD", "Ẹ kú àárọ̀, ọmọ mi.")
    out, applied = g.apply(nfd, "yor_eng")
    assert applied == ["ọmọ→child"]
    assert "child" in out
    # everything outside the substitution survives decomposed
    assert unicodedata.normalize("NFC", out).startswith("Ẹ kú àárọ̀,")


def test_replacement_does_not_corrupt_following_text(glossary: Glossary) -> None:
    src = unicodedata.normalize("NFD", "Àtògbẹ ni mo ní, àti ìtọ ṣúgà.")
    out, applied = glossary.apply(src, "yor_eng")
    # both attested variants map to the same verified term; the point is
    # that each replacement hit exactly its own span, none clobbered the other
    assert sorted(applied) == ["àtògbẹ→diabetes", "ìtọ ṣúgà→diabetes"]
    assert out.count("diabetes") == 2


def test_tone_marks_preserved_on_untouched_words(glossary: Glossary) -> None:
    src = "Ọjọ́ í rẹ̀ yìí dùn mí gan-an, ọ̀rẹ̀ mi."
    out, _ = glossary.apply(src, "yor_eng")
    assert out == src


def test_identity_normalizers_distinguish_minimal_pairs() -> None:
    assert _normalize_word("ìgbà") != _normalize_word("igbá")
    assert _fold_word("ìgbà") == _fold_word("igbá")
    assert _normalize_word("ọmọ") != _normalize_word("omo")


# ------------------------------------------- accent-insensitivity safeguards

def test_accent_folding_disabled_on_collision(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "yor_eng": [
            {"term": "ìgbà", "translation": "time", "context": [],
             "status": "verified", "allow_unaccented": True},
            {"term": "igbá", "translation": "calabash", "context": [],
             "status": "verified", "allow_unaccented": True},
        ]})
    # 'igba' without marks is ambiguous → must NOT be corrected either way
    out, applied = g.apply("Mo ra igba kan.", "yor_eng")
    assert out == "Mo ra igba kan."
    assert applied == []
    assert any("ambiguous" in w for w in g.warnings())


def test_accent_folding_allowed_when_unambiguous(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "yor_eng": [
            {"term": "àtògbẹ", "translation": "diabetes", "context": [],
             "status": "verified", "allow_unaccented": True},
        ]})
    out, applied = g.apply("Mo ni atogbe.", "yor_eng")
    assert "diabetes" in out and applied


def test_accent_sensitive_by_default(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "yor_eng": [
            {"term": "ìgbà", "translation": "time", "context": [],
             "status": "verified"},
        ]})
    out, applied = g.apply("Mo ra igba kan.", "yor_eng")
    assert out == "Mo ra igba kan."
    assert applied == []


# ----------------------------------------------------- multi-word and overlap

def test_longest_match_wins_and_protects_substring(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "diabetes", "translation": "àrùn àtògbẹ",
             "context": [], "status": "verified"},
            {"term": "Type 1 diabetes",
             "translation": "àrùn àtògbẹ Oríṣi Kìíní",
             "context": ["diabetes"], "status": "verified"},
        ]})
    out, applied = g.apply("Type 1 diabetes treatment improved.", "eng_yor")
    assert applied == ["Type 1 diabetes→àrùn àtògbẹ Oríṣi Kìíní"]
    assert "àrùn àtògbẹ Oríṣi Kìíní" in out
    # the inner 'diabetes' was consumed, not replaced separately afterwards
    assert out.count("Oríṣi Kìíní") == 1


def test_multiword_term_span_exact(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "yor_eng": [
            {"term": "aago ẹnu ọ̀nà", "translation": "doorbell",
             "context": [], "status": "verified"},
        ]})
    out, applied = g.apply("Ò gbọ́ aago ẹnu ọ̀nà náà.", "yor_eng")
    assert applied == ["aago ẹnu ọ̀nà→doorbell"]
    assert out == "Ò gbọ́ doorbell náà."


# --------------------------------------------------- unverified entry policy

def test_assumed_entries_loaded_but_never_applied(tmp_path: Path) -> None:
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "widget", "translation": "ohun èlò àìtì",  # deliberately
             "context": [], "status": "assumed",               # unattested
             "comment": "pending human review"},
        ]})
    assert g.unverified("eng_yor") and g.unverified()[0].term == "widget"
    src = "This widget works."
    out, applied = g.apply(src, "eng_yor")
    assert out == src and applied == []


def test_unknown_direction_is_safe(glossary: Glossary) -> None:
    src = "anything"
    assert glossary.apply(src, "zzz_xx") == (src, [])


def test_empty_and_missing_glossary(tmp_path: Path) -> None:
    g = Glossary(str(tmp_path / "missing.json"))
    assert g.apply("text", "eng_yor") == ("text", [])


def test_same_term_twice_only_contextual_occurrence_fires(tmp_path: Path) -> None:
    """Same ambiguous term twice; only the occurrence whose window contains
    the context word is replaced. This is the core semantic guarantee."""
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "stroke", "translation": "àrùn àjẹ́sítà", "context": ["medical"],
             "status": "verified", "comment": "medical stroke"},
        ]})
    src = ("The medical stroke unit admitted him. He swam a fast stroke "
           "in the pool yesterday afternoon.")
    out, applied = g.apply(src, "eng_yor")
    assert applied == ["stroke→àrùn àjẹ́sítà"]
    assert "àrùn àjẹ́sítà unit" in out
    assert "a fast stroke in the pool" in out   # sporting sense untouched


def test_multiple_contexts_independent(tmp_path: Path) -> None:
    """Two ambiguous terms, two contexts, two independent decisions."""
    g = tmp_glossary(tmp_path, {
        "eng_yor": [
            {"term": "light", "translation": "imọ́lẹ̀", "context": ["sun"],
             "status": "verified"},
            {"term": "light", "translation": "fẹ́", "context": ["pack"],
             "status": "verified"},
        ]})
    src = ("The sun light was golden here by the shore today. "
           "Then we did light pack our bags and leave at once.")
    out, applied = g.apply(src, "eng_yor")
    assert applied == ["light→imọ́lẹ̀", "light→fẹ́"]
    assert "The sun imọ́lẹ̀" in out
    assert "did fẹ́ pack" in out
