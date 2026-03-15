from __future__ import annotations

import os
import dotenv
import time
from dataclasses import dataclass
from typing import Any, Optional

from summarizer.evaluator import OpenAIEvaluator
from summarizer.metrics import SummaryMetrics, estimate_cost_usd
from summarizer.telemetry import NoopTelemetrySink, TelemetrySink

dotenv.load_dotenv()

@dataclass(slots=True)
class SummaryResult:
    summary: str
    metrics: SummaryMetrics


class OpenAISummarizer:
    def __init__(
        self,
        client: Any,
        max_output_tokens: int = 500,
        model: Optional[str] = None,
        evaluator: OpenAIEvaluator | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self.client = client
        self.model = model if model else os.getenv("OPENAI_MODEL_SUMMARIZATION", None)
        self.max_output_tokens = max_output_tokens
        self.evaluator = evaluator
        self.telemetry = telemetry or NoopTelemetrySink()

    def summarize(self, text: str) -> SummaryResult:
        context = self.telemetry.start_run(input_text=text, model=self.model)

        started = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise assistant. Summarize the user text in 3-5 sentences."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_output_tokens=self.max_output_tokens,
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 3)

        output_text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) if usage else 0

        metrics = SummaryMetrics(
            duration_ms=duration_ms,
            input_text_length=len(text),
            output_text_length=len(output_text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

        evaluator_score: float | None = None
        if self.evaluator is not None:
            evaluator_score = self.evaluator.evaluate(text=text, summary=output_text)

        self.telemetry.finish_run(
            context=context,
            input_text=text,
            output_text=output_text,
            metrics=metrics,
            evaluator_score=evaluator_score,
        )
        return SummaryResult(summary=output_text, metrics=metrics)
