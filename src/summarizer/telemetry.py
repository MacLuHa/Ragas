from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from summarizer.metrics import SummaryMetrics


@dataclass(slots=True)
class TelemetryRunContext:
    trace: Any | None = None
    generation: Any | None = None


class TelemetrySink(Protocol):
    def start_run(self, input_text: str, model: str) -> TelemetryRunContext: ...

    def finish_run(
        self,
        context: TelemetryRunContext,
        input_text: str,
        output_text: str,
        metrics: SummaryMetrics,
        evaluator_score: float | None = None,
    ) -> None: ...


class NoopTelemetrySink:
    def start_run(self, input_text: str, model: str) -> TelemetryRunContext:
        return TelemetryRunContext()

    def finish_run(
        self,
        context: TelemetryRunContext,
        input_text: str,
        output_text: str,
        metrics: SummaryMetrics,
        evaluator_score: float | None = None,
    ) -> None:
        return None
