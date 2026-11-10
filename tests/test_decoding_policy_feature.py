"""Tests for the decoding-policy module (Agent 4, feature/nllb-optimization).

All tests use fake settings objects and SimpleNamespace mocks — no model
downloads, no real NLLB inference, mirroring the repo's established
mock-injection conventions (see tests/conftest.py).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.decoding_policy import (
    DEFAULT_PROFILE,
    EXPERIMENT_MATRIX,
    PROFILES,
    PRODUCTION_DEFAULT,
    DecodingPolicy,
    build_generate_kwargs,
    experiment_plan_md,
    get_policy,
    load_policy_file,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def fake_settings(**overrides):
    """Stand-in for app.core.config Settings with the production defaults."""
    base = dict(
        translation_num_beams=4,
        translation_max_new_tokens=256,
        translation_length_penalty=1.1,
        translation_max_input_chars=2000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Policy resolution & validation
# ---------------------------------------------------------------------------

def test_default_profile_matches_current_production_parameters() -> None:
    p = get_policy()
    assert p.name == DEFAULT_PROFILE
    assert p.num_beams == 4
    assert p.max_new_tokens == 256
    assert p.length_penalty == 1.1
    # experimental features off by default (conservative)
    assert p.repetition_penalty == 1.0
    assert p.no_repeat_ngram_size == 0
    assert p.early_stopping is None
    assert p.min_new_tokens == 0
    assert not p.dynamic_output_budget
    assert not p.direction_aware_beams


def test_unknown_profile_raises_instead_of_silent_fallback(monkeypatch) -> None:
    monkeypatch.delenv("DECODING_POLICY", raising=False)
    with pytest.raises(ValueError, match="unknown decoding policy 'nope'"):
        get_policy(profile="nope")


def test_resolution_order_env_beats_default_and_arg_beats_env(monkeypatch) -> None:
    monkeypatch.setenv("DECODING_POLICY", "fast_latency")
    assert get_policy().name == "fast_latency"
    assert get_policy(profile="production_default").name == "production_default"


def test_settings_profile_attribute_respected(monkeypatch) -> None:
    monkeypatch.delenv("DECODING_POLICY", raising=False)
    s = SimpleNamespace(decoding_policy_profile="quality_lp12")
    assert get_policy(settings=s).name == "quality_lp12"


@pytest.mark.parametrize("bad", [
    dict(num_beams=0), dict(num_beams=9),
    dict(max_new_tokens=0), dict(max_new_tokens=5000),
    dict(length_penalty=-0.1), dict(length_penalty=2.5),
    dict(repetition_penalty=0.9), dict(repetition_penalty=3.0),
    dict(no_repeat_ngram_size=99),
    dict(min_new_tokens=300),
    dict(dynamic_output_budget=True, budget_input_ratio=0.0),
    dict(direction_aware_beams={"eng->yor": 2}),
])
def test_invalid_policies_rejected(bad) -> None:
    with pytest.raises(ValueError):
        DecodingPolicy(name="bad", **bad)


def test_all_builtin_profiles_validate() -> None:
    assert PROFILES and all(p.name == key for key, p in PROFILES.items())


# ---------------------------------------------------------------------------
# generate-kwargs bridge & backward compatibility
# ---------------------------------------------------------------------------

def test_kwargs_backward_compatible_with_stock_settings() -> None:
    """Default policy + stock settings == exactly the current service params."""
    kw = build_generate_kwargs(PRODUCTION_DEFAULT, fake_settings())
    assert kw == {
        "max_new_tokens": 256,
        "num_beams": 4,
        "length_penalty": 1.1,
    }


def test_kwargs_mirror_settings_operator_overrides_for_default_policy() -> None:
    """If an operator changes settings via .env, default policy follows it."""
    s = fake_settings(translation_num_beams=2, translation_max_new_tokens=300,
                      translation_length_penalty=1.0)
    kw = build_generate_kwargs(PRODUCTION_DEFAULT, s)
    assert kw["num_beams"] == 2
    assert kw["max_new_tokens"] == 300
    assert kw["length_penalty"] == 1.0


def test_kwargs_work_without_settings_object() -> None:
    kw = build_generate_kwargs(PRODUCTION_DEFAULT, None)
    assert kw["num_beams"] == 4 and kw["max_new_tokens"] == 256 and kw["length_penalty"] == 1.1


def test_experimental_policy_is_authoritative_over_settings() -> None:
    s = fake_settings(translation_num_beams=8)  # ambient override must NOT leak
    kw = build_generate_kwargs(PROFILES["fast_latency"], s)
    assert kw["num_beams"] == 2
    assert kw["length_penalty"] == 1.1
    assert kw["max_new_tokens"] == 256


def test_optional_knobs_absent_until_enabled() -> None:
    kw = build_generate_kwargs(PRODUCTION_DEFAULT, fake_settings())
    for absent in ("repetition_penalty", "no_repeat_ngram_size",
                   "early_stopping", "min_new_tokens"):
        assert absent not in kw


def test_enabled_knobs_appear() -> None:
    p = DecodingPolicy(name="x", repetition_penalty=1.15, no_repeat_ngram_size=3,
                       early_stopping=True, min_new_tokens=8)
    kw = build_generate_kwargs(p, fake_settings())
    assert kw["repetition_penalty"] == 1.15
    assert kw["no_repeat_ngram_size"] == 3
    assert kw["early_stopping"] is True
    assert kw["min_new_tokens"] == 8


def test_forced_bos_and_inputs_remain_caller_responsibility() -> None:
    """The bridge returns only decoding args; caller merges its own kwargs."""
    kw = build_generate_kwargs(PRODUCTION_DEFAULT, fake_settings())
    merged = {"forced_bos_token_id": 999, "input_key": object(), **kw}
    assert merged["forced_bos_token_id"] == 999  # nothing clobbered it


# ---------------------------------------------------------------------------
# Dynamic output budget (truncation handling)
# ---------------------------------------------------------------------------

def test_flat_cap_when_dynamic_budget_off() -> None:
    assert PRODUCTION_DEFAULT.effective_max_new_tokens(500) == 256
    assert PRODUCTION_DEFAULT.effective_max_new_tokens(None) == 256


def test_dynamic_budget_scales_and_clamps() -> None:
    p = PROFILES["conservative_guards"]
    assert p.effective_max_new_tokens(20) == p.budget_min          # floor (2*20=40 < 64)
    assert p.effective_max_new_tokens(100) == 200                  # ratio
    assert p.effective_max_new_tokens(10_000) == p.budget_max      # ceiling
    assert p.effective_max_new_tokens(None) == p.max_new_tokens    # unknown input


def test_dynamic_budget_capped_by_max_new_tokens_constraint() -> None:
    p = DecodingPolicy(name="y", dynamic_output_budget=True,
                       max_new_tokens=128, budget_min=64, budget_max=400)
    assert p.budget_max <= p.max_new_tokens * 4  # validation invariant holds


def test_direction_aware_beams() -> None:
    p = PROFILES["direction_aware_beams"]
    assert p.effective_num_beams("yor_eng") == 2
    assert p.effective_num_beams("eng_yor") == 4
    assert p.effective_num_beams(None) == 4
    kw = build_generate_kwargs(p, fake_settings(), direction="yor_eng")
    assert kw["num_beams"] == 2


# ---------------------------------------------------------------------------
# Integration-shape smoke test with mocked model/tokenizer (no inference)
# ---------------------------------------------------------------------------

class FakeTokenizer:
    src_lang = "eng_Latn"

    def convert_tokens_to_ids(self, token: str) -> int:
        return 999

    def batch_decode(self, ids, **kw):
        return ["fake translation"]


def test_kwargs_feed_a_mock_generate_without_error() -> None:
    """Demonstrates the exact contract translation.py would consume."""
    calls = {}

    class FakeModel:
        def generate(self, **kwargs):
            calls.update(kwargs)
            return [[1, 2]]

    kw = build_generate_kwargs(PRODUCTION_DEFAULT, fake_settings())
    FakeModel().generate(forced_bos_token_id=999, **kw)
    assert calls["forced_bos_token_id"] == 999
    assert calls["num_beams"] == 4


# ---------------------------------------------------------------------------
# Experiment plan integrity (definitions only — no results claimed)
# ---------------------------------------------------------------------------

def test_experiment_matrix_wellformed_and_non_empty() -> None:
    ids = [e["id"] for e in EXPERIMENT_MATRIX]
    assert len(EXPERIMENT_MATRIX) >= 3
    assert len(ids) == len(set(ids))
    for e in EXPERIMENT_MATRIX:
        assert e["policy"] in PROFILES
        assert e["question"] and e["pass_criteria"]


def test_experiment_plan_markdown_renders_all_rows() -> None:
    md = experiment_plan_md()
    for e in EXPERIMENT_MATRIX:
        assert e["id"] in md


def test_no_builtin_profile_claims_unmeasured_quality_gains() -> None:
    """Guard against inventing results: profiles cite only existing artifacts."""
    banned = ["improves chrF", "outperforms baseline by", "state of the art"]
    for p in PROFILES.values():
        low = p.notes.lower()
        for b in banned:
            assert b not in low


def test_load_policy_file_roundtrip(tmp_path) -> None:
    import json
    f = tmp_path / "custom.json"
    f.write_text(json.dumps({"name": "custom_test", "num_beams": 3}))
    p = load_policy_file(f)
    assert p.num_beams == 3


def test_load_policy_file_rejects_unknown_keys_and_missing_name(tmp_path) -> None:
    import json
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "bogus_key": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_policy_file(bad)
    noname = tmp_path / "noname.json"
    noname.write_text(json.dumps({"num_beams": 3}))
    with pytest.raises(ValueError, match="missing required 'name'"):
        load_policy_file(noname)


def test_with_overrides_derives_named_variant() -> None:
    v = PRODUCTION_DEFAULT.with_overrides(name="derived", num_beams=6)
    assert v.name == "derived" and v.num_beams == 6
    assert PRODUCTION_DEFAULT.num_beams == 4  # original untouched (frozen)
