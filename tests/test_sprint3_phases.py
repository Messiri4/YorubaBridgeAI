"""Sprint 3 tests: PHASE 1 (classifier audit), PHASE 2 (tracing + bounded
retry), PHASE 3 (meaning-error detection heuristics).

Production changes are minimal and gated; diagnostics are opt-in.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.model_loader import LoadedModel
from app.services.translation import TranslationService


# ============================ PHASE 1 ========================================

def test_numword_lexicon_no_cross_value_collisions() -> None:
    from scripts.sprint2_failure_analysis import _fold, _NUM_WORDS
    # build value -> folded-words map and assert no word maps to two values
    owner: dict[str, int] = {}
    for value, words in _NUM_WORDS.items():
        for w in words:
            f = _fold(w)
            assert f not in owner or owner[f] == value, \
                f"collision: {f!r} maps to {owner.get(f)} and {value}"
            owner[f] = value


def test_ogorin_80_does_not_verify_ogorun_100() -> None:
    from scripts.sprint2_failure_analysis import classify_row
    st, _ = classify_row("Speed is 80 km/h.", "Ìyára kàn ọgọ́rùn.")  # 100-word
    assert st == "dropped"  # a 100-rendering must not satisfy source 80


def test_whole_token_match_blocks_substring_verification() -> None:
    from scripts.sprint2_failure_analysis import classify_row
    # 70 ('aadorin') embedded inside the 76-compound must not verify 90
    st, _ = classify_row("About 90 persons.", "Ní mẹ́rìndínláàádọ́rin.")
    assert st == "dropped"


def test_legitimate_spellings_still_verified() -> None:
    from scripts.sprint2_failure_analysis import classify_row
    assert classify_row("He scored 4.", "Ó gba mẹ́rin.")[0] == "rendered_as_words"
    assert classify_row("About 50.", "Ní báyìí àádọ́ta.")[0] == "rendered_as_words"
    # source without digits is 'no_numbers' by design (digit-oriented check);
    # spelled-source equivalence is out of scope for this detector
    assert classify_row("19 left.", "Mókàndínlógún kù.")[0] == "rendered_as_words"


def test_approximate_counts_not_reported_as_verified() -> None:
    """The analysis doc must not present heuristic classification counts as
    verified error rates."""
    doc = Path("docs/sprint2_failure_analysis.md").read_text(encoding="utf-8")
    assert "verified error rate" not in doc.lower()
    assert "pending human linguistic review" in doc.lower()


# ============================ PHASE 2 ========================================

def _service(outputs: list[str], max_retries: int) -> tuple[TranslationService, MagicMock]:
    """outputs[i] is returned for the i-th model.generate() CALL (the true
    generation boundary), immune to internal batch_decode call counts."""
    get_settings.cache_clear()
    s = get_settings()
    s.numeric_protection_enabled = True
    s.numeric_protection_max_retries = max_retries
    tok = MagicMock()
    tok.src_lang = "eng_Latn"
    tok.return_value = {"input_ids": [[1]]}
    tok.convert_tokens_to_ids.return_value = 1
    m = MagicMock()
    gen = iter(outputs)

    def _decode(ids, **kw):
        # Serve the CURRENT generation's output; advance only when the
        # harness is told to (via next_output below).
        return [state["cur"]]

    state = {"cur": outputs[0]}

    def _generate(**kw):
        # advance to the next scripted output for each NEW generation after
        # the first (the first consumes outputs[0] directly)
        if state.get("started"):
            state["cur"] = next(gen, state["cur"])
        state["started"] = True
        return [[1]]

    m.generate.side_effect = _generate
    tok.batch_decode.side_effect = _decode
    return TranslationService(LoadedModel(model=m, tokenizer=tok, device="cpu")), m


def test_bounded_retry_succeeds_on_second_protected_attempt() -> None:
    # gen1 drops the sentinel; gen2 (bounded protected retry) delivers it —
    # the plain regeneration is never needed
    svc, _ = _service(["Ó wà.", "Ó wà ní 7–2."], max_retries=1)
    r = svc.translate("The score was 7–2.", "eng", "yor")
    assert "7–2" in r.translated_text and "@@" not in r.translated_text


def test_bounded_retry_exhausts_then_plain_fallback() -> None:
    # outputs: gen1 (protected, fails) -> gen2 (protected retry, fails) ->
    # gen3 (plain fallback, carries the number)
    svc, _ = _service(["Ó wà kú @@0@@.", "Ó wà kú @@0@@.", "Ó wà ní 7–2."],
                      max_retries=1)
    r = svc.translate("The score was 7–2.", "eng", "yor")
    assert "7–2" in r.translated_text and "@@" not in r.translated_text


def test_max_retries_zero_is_legacy_one_shot() -> None:
    # gen1 (protected, fails) -> gen2 (plain fallback carries number)
    svc, _ = _service(["Ó wà kú @@0@@.", "Ó wà ní 7–2."], max_retries=0)
    r = svc.translate("The score was 7–2.", "eng", "yor")
    assert "7–2" in r.translated_text and "@@" not in r.translated_text


def test_diagnostic_trace_records_stages() -> None:
    import app.services.diagnostic_trace as dt
    svc, _ = _service(["Ó wà ní 7–2."], max_retries=0)
    dt._ENABLED_FILE = True  # enable tracing for this test only
    try:
        dt.reset_traces()
        svc.translate("The score was 7–2.", "eng", "yor")
        traces = dt.get_traces()
        assert traces, "expected at least one trace"
        names = [s["name"] for s in traces[-1]["stages"]]
        assert "protect" in names and "generate_protected" in names \
            and "restore" in names and "finish" in names
    finally:
        dt._ENABLED_FILE = None
        dt.reset_traces()


def test_diagnostic_trace_inert_by_default(monkeypatch) -> None:
    import app.services.diagnostic_trace as dt
    monkeypatch.delenv("YB_TRANSLATION_TRACE", raising=False)
    monkeypatch.delenv("YB_TRANSLATION_TRACE_ECHO", raising=False)
    svc, _ = _service(["Ó wà ní 7–2."], max_retries=0)
    n_before = len(dt.get_traces())
    svc.translate("The score was 7–2.", "eng", "yor")
    assert len(dt.get_traces()) == n_before  # nothing recorded


# ============================ PHASE 3 ========================================

def test_negation_flip_detection_both_directions() -> None:
    from scripts.meaning_diagnostics import detect_negation_flip
    # yor->eng: prediction negates what the source affirms
    assert detect_negation_flip(
        "Ó ń rò.", "It is not raining.") == "negation_flip_suspected"
    assert detect_negation_flip(
        "Ó ń rò.", "It is raining.") == "ok"
    # eng->yor: prediction inserts ko/ò (negation) absent in source
    assert detect_negation_flip(
        "The door is open.", "Ilẹ̀kùun náà kìí ṣí.") == "negation_flip_suspected"
    assert detect_negation_flip(
        "The door is open.", "Ilẹ̀kùun náà ṣí.") == "ok"


def test_omission_detection_names_numbers_clauses() -> None:
    from scripts.meaning_diagnostics import detect_omissions
    res = detect_omissions(
        "President Kumba sent 500 troops to Lagos.",
        "Aláṣẹ Kumba ranṣẹ́ ọmọ-ogun sí ìlú Lósí.")
    assert "omitted_number:500" in res
    assert any("omitted_entity" in r for r in res)


def test_repetition_degeneration_detection() -> None:
    from scripts.meaning_diagnostics import detect_degeneration
    assert detect_degeneration("A a a a a a a a a b c.") == "repetition_loop"
    assert detect_degeneration("Normal sentence with words.") == "ok"


def test_heuristic_vs_confirmed_labeling() -> None:
    from scripts.meaning_diagnostics import classify_findings
    findings = [
        {"kind": "negation_flip", "confidence": "heuristic"},
        {"kind": "omitted_number", "detail": "500", "in_reference": True},
    ]
    out = classify_findings(findings)
    assert out[0]["status"] == "heuristic_warning"
    # an omission verifiable against a reference is a confirmed-content check,
    # still labeled for review unless human-confirmed
    assert out[1]["status"] == "requires_human_review"
