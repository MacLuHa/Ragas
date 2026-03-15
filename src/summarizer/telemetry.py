from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from summarizer.metrics import SummaryMetrics

import dotenv
import os

dotenv.load_dotenv()


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
        #evaluator_score: float | None = None,
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
        #evaluator_score: float | None = None,
    ) -> None:
        return None


class LangfuseTelemetrySink:
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = None,
        httpx_client: Any | None = None,
    ) -> None:
        from langfuse import Langfuse
        if not host:
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            httpx_client=httpx_client,
        )

    def start_run(self, input_text: str, model: str) -> TelemetryRunContext:
        span = self.client.start_span(
            name="text_summarization",
            input={"text": input_text, "text_length": len(input_text)},
            metadata={"model": model},
        )
        generation = span.start_generation(
            name="openai_summarization",
            model=model,
            input=input_text,
            metadata={"input_text_length": len(input_text)},
        )
        return TelemetryRunContext(trace=span, generation=generation)


    def finish_run(
        self,
        context: TelemetryRunContext,
        input_text: str,
        output_text: str,
        metrics: SummaryMetrics,
        #evaluator_score: float | None = None,
    ) -> None:
        if context.generation is not None:
            context.generation.update(
                input=input_text,
                output=output_text,
                usage_details={
                    "prompt_tokens": metrics.input_tokens,
                    "completion_tokens": metrics.output_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                cost_details={"total_cost": metrics.estimated_cost_usd},
                metadata={
                    "duration_ms": metrics.duration_ms,
                    "estimated_cost_usd": metrics.estimated_cost_usd,
                    "input_text_length": metrics.input_text_length,
                    "output_text_length": metrics.output_text_length,
                },
            )
            #if evaluator_score is not None:
             #   context.generation.score(
              #      name="custom_evaluator",
               #     value=float(evaluator_score),
                #    data_type="NUMERIC",
                #)
            context.generation.end()

        if context.trace is not None:
            context.trace.update(
                output={"summary": output_text, "metrics": metrics.to_dict()},
            )
            context.trace.end()

        self.client.flush()
