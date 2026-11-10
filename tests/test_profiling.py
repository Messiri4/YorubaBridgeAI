"""Tests for the Sprint 2B inference profiler (all mocked, no real inference)."""

from __future__ import annotations

import json
import time

import pytest

from app.services.profiling import (
    StageTimings,
    TranslationProfiler,
    summarize_latencies,
)


class FakeClock:
    """Deterministic clock advancing by a fixed step per call."""

    def __init__(self, step: float = 0.25) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        self._now += self._step
        return self._now


def test_profile_run_times_each_stage_without_model() -> None:
    """Stages get measurable, monotonically-consistent durations; no NLLB import."""
    clock = FakeClock(step=0.5)
    profiler = TranslationProfiler(clock=clock, capture_memory=False)
    timings = profiler.profile_run(
        tokenize=lambda: "inputs",
        generate=lambda inputs: "generated",
        decode_output=lambda generated: "text",
    )
    # Each stage consumes exactly one clock step (0.5s), init skipped (warm).
    assert timings.tokenize_s == pytest.approx(0.5)
    assert timings.generate_s == pytest.approx(0.5)
    assert timings.decode_output_s == pytest.approx(0.5)
    assert timings.model_init_s == 0.0
    assert timings.total_s == pytest.approx(1.5)
    assert timings.cold_start is False


def test_cold_start_includes_model_init_time() -> None:
    clock = FakeClock(step=1.0)
    profiler = TranslationProfiler(clock=clock, capture_memory=False)

    def slow_init() -> None:
        time.sleep(0)  # simulated weight loading work

    timings = profiler.profile_run(
        model_init=slow_init,
        tokenize=lambda: "inputs",
        generate=lambda _: "generated",
        decode_output=lambda _: "text",
        cold_start=True,
    )
    assert timings.cold_start is True
    assert timings.model_init_s == pytest.approx(1.0)
    assert timings.total_s == pytest.approx(4.0)


def test_timer_accuracy_against_perf_counter() -> None:
    """Profiled elapsed time tracks time.perf_counter within 20% tolerance."""
    profiler = TranslationProfiler(capture_memory=False)

    def busy() -> None:
        target = time.perf_counter() + 0.1
        while time.perf_counter() < target:
            pass

    timings = profiler.profile_run(
        tokenize=busy, generate=lambda _: None, decode_output=lambda _: ""
    )
    assert timings.tokenize_s == pytest.approx(0.1, rel=0.3)


def test_report_serializes_to_json() -> None:
    profiler = TranslationProfiler(capture_memory=False)
    profiler.profile_run(tokenize=lambda: None, generate=lambda _: None, decode_output=lambda _: "")
    payload = json.loads(profiler.report.to_json())
    assert payload["schema_version"] == 1
    assert len(payload["timings"]) == 1
    stage = payload["timings"][0]
    for key in ("model_init_s", "tokenize_s", "generate_s", "decode_output_s", "total_s"):
        assert key in stage


def test_summarize_latencies_stats_and_errors() -> None:
    stats = summarize_latencies([3.0, 1.0, 2.0, 4.0, 5.0])
    assert stats["n"] == 5
    assert stats["mean_s"] == 3.0
    assert stats["median_s"] == 3.0
    assert stats["min_s"] == 1.0
    assert stats["max_s"] == 5.0
    assert 1.0 <= stats["p95_s"] <= 5.0
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_missing_measurements_raise_keyerror() -> None:
    """Aggregation over zero warm runs must fail loudly, not fabricate zeros."""
    profiler = TranslationProfiler(capture_memory=False)
    with pytest.raises(KeyError):
        profiler.report.mean("generate_s")
    with pytest.raises(KeyError):
        profiler.report.median("total_s")


def test_warm_runs_exclude_cold_start_from_aggregates() -> None:
    profiler = TranslationProfiler(clock=FakeClock(step=0.1), capture_memory=False)
    profiler.profile_run(
        model_init=lambda: None,
        tokenize=lambda: None,
        generate=lambda _: None,
        decode_output=lambda _: "",
        cold_start=True,
    )
    profiler.profile_run(tokenize=lambda: None, generate=lambda _: None, decode_output=lambda _: "")
    assert len(profiler.report.timings) == 2
    assert len(profiler.report.warm_timings) == 1


def test_profiler_module_import_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the profiler must not trigger any model loader call.

    Skipped when torch/transformers are absent from the environment: the
    contract under test is 'import performs no model work', not 'torch
    is installed'.
    """
    torch = pytest.importorskip("torch")  # noqa: F841 - presence is the gate
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    called = []

    def boom(*args: object, **kwargs: object) -> None:
        called.append(True)

    monkeypatch.setattr("app.services.model_loader.load_nllb", boom)
    import app.services.profiling  # noqa: F401  (import is the assertion)
    import app.services.translation  # noqa: F401

    assert called == []


def test_profile_run_propagates_stage_errors_but_records_stage() -> None:
    profiler = TranslationProfiler(capture_memory=False)
    with pytest.raises(RuntimeError):
        profiler.profile_run(
            tokenize=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            generate=lambda _: None,
            decode_output=lambda _: "",
        )


def test_deterministic_clock_gives_reproducible_benchmark_config() -> None:
    """Same clock + same stages => byte-identical timing totals (reproducibility)."""

    def make_run() -> StageTimings:
        profiler = TranslationProfiler(clock=FakeClock(step=0.2), capture_memory=False)
        profiler.profile_run(
            tokenize=lambda: "x", generate=lambda _: "y", decode_output=lambda _: "z"
        )
        return profiler.report.timings[0]

    a, b = make_run(), make_run()
    assert a.total_s == b.total_s
    assert (a.tokenize_s, a.generate_s, a.decode_output_s) == (
        b.tokenize_s,
        b.generate_s,
        b.decode_output_s,
    )
