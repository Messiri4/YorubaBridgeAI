"""Reusable inference profiler for the translation pipeline (Sprint 2B).

Purpose: measure WHERE time and memory go in a translation request
(model init, tokenization, generation, detokenization) and produce a
JSON-serializable :class:`ProfileReport` for comparison runs.

Design constraints (per Sprint 2B):
- Importing this module must NEVER load NLLB or any model. All heavy
  work happens behind explicitly opt-in methods/functions.
- Instrumentation is wall-clock (`time.perf_counter`) plus optional peak
  RSS sampling when available; it never changes production decoding
  configuration.
- Unit tests use fake timers / mock models; no real inference runs here.
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _peak_rss_mb() -> float | None:
    """Best-effort peak RSS in MiB; None when the platform lacks resource."""
    try:
        import resource

        # macOS reports ru_maxrss in bytes, Linux in KiB.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 * 1024) if rss > 10_000_000 else rss / 1024, 1)
    except Exception:  # noqa: BLE001 - profiling must never break callers
        return None


class _Clock(Protocol):
    """Minimal clock interface so tests can inject a deterministic one."""

    def __call__(self) -> float: ...


@dataclass
class StageTimings:
    """Wall-clock durations (seconds) for one instrumented pipeline run."""

    model_init_s: float = 0.0
    tokenize_s: float = 0.0
    generate_s: float = 0.0
    decode_output_s: float = 0.0
    total_s: float = 0.0
    cold_start: bool = True
    peak_rss_mb: float | None = None


@dataclass
class ProfileReport:
    """Aggregated, JSON-serializable profiler output."""

    schema_version: int = 1
    timings: list[StageTimings] = field(default_factory=list)

    @property
    def warm_timings(self) -> list[StageTimings]:
        return [t for t in self.timings if not t.cold_start]

    def mean(self, attribute: str) -> float:
        """Mean of one timing field across warm runs only (cold runs excluded)."""
        values = [getattr(t, attribute) for t in self.warm_timings]
        if not values:
            raise KeyError(f"no warm measurements recorded for '{attribute}'")
        return statistics.mean(values)  # type: ignore[arg-type, return-value]

    def median(self, attribute: str) -> float:
        """Median of one timing field across warm runs only."""
        values = [getattr(t, attribute) for t in self.warm_timings]
        if not values:
            raise KeyError(f"no warm measurements recorded for '{attribute}'")
        return statistics.median(values)  # type: ignore[arg-type, return-value]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class TranslationProfiler:
    """Instruments a translate-like pipeline without owning a model.

    The profiler takes *callables* for each pipeline stage, so callers
    decide whether the stage talks to the real NLLB model or to a mock.
    Constructing this class performs no model work whatsoever.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        capture_memory: bool = True,
    ) -> None:
        self._clock = clock
        self._capture_memory = capture_memory
        self._report = ProfileReport()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _measure(self, fn: Callable[[], Any]) -> tuple[float, Any]:
        """Run `fn` once and return (elapsed_seconds, result)."""
        start = self._clock()
        try:
            result = fn()
        except BaseException:
            # A failing stage still gets accounted for as elapsed time;
            # the exception propagates so callers see real errors.
            elapsed = self._clock() - start
            logger.debug("profiler stage raised after %.4fs", elapsed)
            raise
        return self._clock() - start, result

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def profile_run(
        self,
        *,
        model_init: Callable[[], Any] | None = None,
        tokenize: Callable[[], Any],
        generate: Callable[[Any], Any],
        decode_output: Callable[[Any], str],
        cold_start: bool = False,
    ) -> StageTimings:
        """Time one full pipeline pass.

        Args:
            model_init: optional stage; skipped (recorded as 0.0) when the
                model is already loaded (warm run) or not provided.
            tokenize: produce model inputs from raw text.
            generate: run autoregressive generation on the tokenized input.
            decode_output: convert generated ids to text.
            cold_start: mark this run as a cold start (first pass after init).

        Returns:
            The StageTimings for this pass (also accumulated in the report).
        """
        timings = StageTimings(cold_start=cold_start)
        try:
            if model_init is not None:
                timings.model_init_s, _ = self._measure(model_init)

            timings.tokenize_s, inputs = self._measure(tokenize)
            timings.generate_s, generated = self._measure(lambda: generate(inputs))
            timings.decode_output_s, _text = self._measure(lambda: decode_output(generated))
        finally:
            timings.total_s = (
                timings.model_init_s
                + timings.tokenize_s
                + timings.generate_s
                + timings.decode_output_s
            )
            if self._capture_memory:
                timings.peak_rss_mb = _peak_rss_mb()
            self._report.timings.append(timings)
        return timings

    @property
    def report(self) -> ProfileReport:
        return self._report

    def reset(self) -> None:
        self._report = ProfileReport()


def summarize_latencies(seconds: list[float]) -> dict[str, float]:
    """Summary statistics for a list of latency samples, in seconds.

    Raises:
        ValueError: if the sample list is empty.
    """
    if not seconds:
        raise ValueError("no latency samples provided")
    ordered = sorted(seconds)

    def pct(p: float) -> float:
        rank = max(1, round(0.01 * p * len(ordered)))
        return ordered[rank - 1]

    return {
        "n": float(len(seconds)),
        "mean_s": round(statistics.mean(seconds), 3),
        "median_s": round(statistics.median(seconds), 3),
        "p95_s": round(pct(95), 3),
        "min_s": round(ordered[0], 3),
        "max_s": round(ordered[-1], 3),
    }


def collect_system_context() -> dict[str, Any]:
    """Static environment facts relevant to interpreting profiles."""
    context: dict[str, Any] = {"platform_peak_rss_mb": _peak_rss_mb()}
    try:
        import platform

        context["machine"] = platform.machine()
        context["python"] = platform.python_version()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        context["torch"] = torch.__version__
        context["torch_num_threads"] = torch.get_num_threads()
        context["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        context["torch"] = None
    return context


def maybe_collect_gc() -> None:
    """Explicit GC before benchmark passes to reduce jitter (opt-in)."""
    gc.collect()
