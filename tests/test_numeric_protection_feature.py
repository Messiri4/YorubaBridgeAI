"""Focused tests for the numeric-protection reliability feature.

Covers: repeated numbers, negative numbers, decimals, percentages, times,
scores, currency, Unicode punctuation, placeholder collisions, nested
matches, positional ordering, structured failures and the reusable
unprotected-fallback helper.

No model is loaded: these are pure unit tests against the protection
module's public interface.
"""
from __future__ import annotations

import re

import pytest

from app.services.protection import (
    ProtectionResult,
    RestorationFailure,
    RestorationOutcome,
    SENTINEL_RE,
    protect,
    restore,
    restore_outcome,
    verify_placeholder_structure,
    with_unprotected_fallback,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _indexes(text: str) -> list[str]:
    return SENTINEL_RE.findall(text)


def roundtrip_is_identity(source: str) -> ProtectionResult:
    """protect() then simulate the model copying sentinels verbatim."""
    protection = protect(source)
    model_copies_verbatim = protection.protected_text
    restored, ok, reason = restore(model_copies_verbatim, protection)
    assert ok, reason
    assert restored == source
    return protection


# ---------------------------------------------------------------------------
# basic expression coverage
# ---------------------------------------------------------------------------

def test_repeated_same_number_gets_distinct_sentinels():
    p = roundtrip_is_identity("I have 3 apples and 3 apples")
    assert p.protected_text == "I have @@0@@ apples and @@1@@ apples"
    assert p.placeholders == {"@@0@@": "3", "@@1@@": "3"}


def test_negative_number_is_protected_including_sign():
    p = protect("temp is -12.5C now")
    assert "-12.5" in p.placeholders.values()
    assert re.search(r"-\d", p.protected_text) is None


def test_decimals_and_thousands_separators():
    p = protect("1,000.50 and 2,000")
    assert p.n == 2
    assert "1,000.50" in p.placeholders.values()
    assert "2,000" in p.placeholders.values()


def test_percentage_protected():
    p = protect("about 45% of 100 people, or 50 %")
    assert "45%" in p.placeholders.values()
    assert "50 %" in p.placeholders.values()


def test_time_protected_with_meridiem():
    p = roundtrip_is_identity("meeting at 11:29 pm, call at 07:05")
    assert p.n == 2
    assert "11:29 pm" in p.placeholders.values()
    assert "07:05" in p.placeholders.values()


def test_score_range_protected_including_unicode_dash():
    p = protect("won 7–2, lost 3 - 1")
    assert "7–2" in p.placeholders.values()
    assert "3 - 1" in p.placeholders.values()


def test_currency_symbols_and_names():
    p = protect("$1,000, ₦5,000 and 500 naira")
    assert "$1,000" in p.placeholders.values()
    assert "₦5,000" in p.placeholders.values()
    assert "500 naira" in p.placeholders.values()


def test_email_and_url_protected():
    p = protect("mail bob@example.com or see www.example.org/docs")
    assert "bob@example.com" in p.placeholders.values()
    assert "www.example.org/docs" in p.placeholders.values()


def test_iso_and_dotted_dates():
    p = protect("from 2019-06-04 until 04.06.2019")
    assert "2019-06-04" in p.placeholders.values()
    assert "04.06.2019" in p.placeholders.values()


# ---------------------------------------------------------------------------
# collision / nesting / ordering
# ---------------------------------------------------------------------------

def test_no_nested_sentinels_for_high_numbers():
    p = protect("ids 1234 and 56789")
    assert p.n == 2
    # no sentinel index was produced from digits inside another sentinel
    assert p.protected_text == "ids @@0@@ and @@1@@"
    assert set(p.placeholders) == {"@@0@@", "@@1@@"}


def test_protect_is_idempotent():
    first = protect("3 apples, 11:29 pm")
    second = protect(first.protected_text)
    assert second.protected_text == first.protected_text
    assert second.n == first.n


def test_sentinels_are_numbered_by_text_position():
    p = protect("3 and 11:29 pm and 4")  # time pattern is high priority
    assert p.protected_text == "@@0@@ and @@1@@ and @@2@@"
    assert p.placeholders["@@1@@"] == "11:29 pm"


def test_user_placeholder_like_text_is_not_destroyed():
    source = "literal @@3@@ token plus 5 numbers"
    p = protect(source)
    # The user's token is CLAIMED: rewritten into a fresh sequential
    # sentinel whose stored original is the literal "@@3@@" itself, so
    # the protected text contains only sequentially issued sentinels.
    assert "@@3@@" not in p.protected_text
    assert p.placeholders["@@0@@"] == "@@3@@"
    assert "5" not in p.protected_text  # the real number is protected
    # Roundtrip restores both the number and the user's literal token.
    restored, ok, _ = restore(p.protected_text, p)
    assert ok
    assert restored == source


def test_user_placeholder_collision_with_generated_index():
    # user text already uses @@0@@; a generated sentinel must not merge
    source = "@@0@@ and 7"
    p = protect(source)
    assert p.protected_text.count("@@") == 4  # two sentinels, no merge
    assert p.placeholders["@@0@@"] == "@@0@@"  # user token, verbatim
    assert p.placeholders["@@1@@"] == "7"
    restored, ok, _ = restore(p.protected_text, p)
    assert ok
    assert restored == source


def test_restore_only_substitutes_issued_sentinels():
    p = protect("value 42")
    # a sentinel protect() never issued must not be silently substituted
    r = restore_outcome("@@0@@ and @@5@@", p)
    assert not r.ok
    assert r.failure.kind == "unknown_sentinel"
    assert "@@5@@" in r.failure.unknown


def test_verify_wrapper_backwards_compatible():
    ok, reason = verify_placeholder_structure("@@0@@", 1)
    assert ok and reason == "ok"
    ok, reason = verify_placeholder_structure("@@0@@", 2)
    assert not ok and "@@1@@" in reason  # names the missing sentinel


# ---------------------------------------------------------------------------
# structured failure paths
# ---------------------------------------------------------------------------

def _two_number_protection() -> ProtectionResult:
    return protect("two numbers 5 and 10")


def test_missing_sentinel_failure_is_structured():
    p = _two_number_protection()
    broken = p.protected_text.replace("@@1@@", "", 1)
    r = restore_outcome(broken, p)
    assert not r.ok
    assert r.text == broken  # safe fallback: original output returned
    f = r.failure
    assert isinstance(f, RestorationFailure)
    assert f.kind == "missing"
    assert "@@1@@" in f.missing
    assert f.expected == 2 and f.found == 1


def test_duplicated_sentinel_failure_is_structured():
    p = protect("one number 5")
    r = restore_outcome("@@0@@ @@0@@", p)
    assert not r.ok
    assert r.failure.kind == "duplicated"
    assert "@@0@@" in r.failure.duplicated


def test_missing_and_duplicated_classified_jointly():
    p = _two_number_protection()
    broken = p.protected_text.replace("@@1@@", "@@0@@")
    r = restore_outcome(broken, p)
    assert not r.ok
    assert r.failure.kind == "missing_and_duplicated"
    assert "@@1@@" in r.failure.missing
    assert "@@0@@" in r.failure.duplicated


def test_restore_tuple_api_reports_reason():
    p = protect("value 42")
    text, ok, reason = restore("@@1@@", p)
    assert not ok
    assert text == "@@1@@"  # unchanged on failure
    assert "missing" in reason and "@@0@@" in reason


def test_nothing_to_restore_is_trivial_success():
    p = protect("no numbers here")
    assert p.n == 0
    r = restore_outcome("any output", p)
    assert r.ok and r.text == "any output"


# ---------------------------------------------------------------------------
# unicode punctuation adjacent to numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    "«2» items",
    "— 3 —",
    "(4), [5], {6}",
    "“7”…“8”",
    "price: €3,50; total: 10%",
])
def test_unicode_punctuation_neighbors(source):
    p = roundtrip_is_identity(source)
    assert p.n >= 1


def test_unicode_dash_range_not_confused_with_time():
    p = protect("2:1 and 7–2 at 11:29")
    assert "2:1" in p.placeholders.values()
    assert "7–2" in p.placeholders.values()
    assert "11:29" in p.placeholders.values()


# ---------------------------------------------------------------------------
# reusable fallback helper
# ---------------------------------------------------------------------------

class _FakeProtectedResult:
    def __init__(self, outcome: RestorationOutcome):
        self.payload = outcome


def test_fallback_helper_not_triggered_on_success():
    outcome = RestorationOutcome(text="ok", ok=True)

    def protected(**kw):
        return _FakeProtectedResult(outcome)

    def plain(**kw):
        raise AssertionError("plain path must not run on success")

    result, used_fallback, failure = with_unprotected_fallback(
        protected, plain, text="x")
    # Success returns the caller's protected result unchanged (the wrapper
    # object here; its .payload carries the RestorationOutcome).
    assert result is not plain
    assert result.payload is outcome
    assert used_fallback is False
    assert failure is None


def test_fallback_helper_retries_unprotected_on_failure():
    bad = RestorationOutcome(
        text="raw", ok=False,
        failure=RestorationFailure(kind="missing", detail="x", expected=1, found=0))

    def protected(**kw):
        assert kw["protect_numerics"] is True
        return _FakeProtectedResult(bad)

    def plain(**kw):
        assert kw.get("protect_numerics") is False
        return "plain-translation"

    result, used_fallback, failure = with_unprotected_fallback(
        protected, plain, text="x", protect_numerics=True)
    assert result == "plain-translation"
    assert used_fallback is True
    assert failure.kind == "missing"


def test_fallback_helper_respects_attempt_fallback_gate():
    bad = RestorationOutcome(
        text="raw", ok=False,
        failure=RestorationFailure(kind="count_mismatch", detail="x"))

    def protected():
        return RestorationOutcome(text="raw", ok=False, failure=bad.failure)

    def plain():
        raise AssertionError("gate said no retry")

    result, used_fallback, failure = with_unprotected_fallback(
        protected, plain, attempt_fallback=lambda outcome, text: False)
    assert used_fallback is False
    assert failure.kind == "count_mismatch"


def test_fallback_helper_invokes_on_failure_hook():
    seen = []
    bad = RestorationOutcome(
        text="raw", ok=False,
        failure=RestorationFailure(kind="missing", detail="d"))

    def protected():
        return RestorationOutcome(text="raw", ok=False, failure=bad.failure)

    def plain():
        return "plain"

    with_unprotected_fallback(
        protected, plain,
        on_failure=lambda f, t: seen.append((f.kind, t)))
    assert seen == [("missing", "raw")]
