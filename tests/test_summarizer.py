from __future__ import annotations

from types import SimpleNamespace

from summarizer.service import OpenAISummarizer
from summarizer.telemetry import TelemetryRunContext


class FakeResponsesAPI:
    def __init__(self, response):
        self._response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self._response


class FakeOpenAIClient:
    def __init__(self, response):
        self.responses = FakeResponsesAPI(response=response)


class SpyTelemetry:
    def __init__(self):
        self.started = None
        self.finished = None

    def start_run(self, input_text: str, model: str):
        self.started = {"input_text": input_text, "model": model}
        return TelemetryRunContext()

    def finish_run(
        self,
        context,
        input_text: str,
        output_text: str,
        metrics,
        evaluator_score: float | None = None,
    ):
        self.finished = {
            "input_text": input_text,
            "output_text": output_text,
            "metrics": metrics,
            "evaluator_score": evaluator_score,
        }


def test_collects_required_metrics_and_io():
    response = SimpleNamespace(
        output_text="Короткое резюме исходного текста.",
        usage=SimpleNamespace(input_tokens=120, output_tokens=24, total_tokens=144),
    )
    client = FakeOpenAIClient(response=response)
    telemetry = SpyTelemetry()
    service = OpenAISummarizer(client=client, model="gpt-4o-mini", telemetry=telemetry)

    source_text = "Очень длинный исходный текст для суммаризации."
    result = service.summarize(source_text)

    assert result.summary == "Короткое резюме исходного текста."
    assert result.metrics.duration_ms >= 0
    assert result.metrics.input_text_length == len(source_text)
    assert result.metrics.output_text_length == len(result.summary)
    assert result.metrics.input_tokens == 120
    assert result.metrics.output_tokens == 24
    assert result.metrics.total_tokens == 144
    assert result.metrics.estimated_cost_usd > 0
    assert telemetry.started["input_text"] == source_text
    assert telemetry.finished["output_text"] == result.summary


def test_unknown_model_returns_zero_cost():
    response = SimpleNamespace(
        output_text="summary",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    client = FakeOpenAIClient(response=response)
    service = OpenAISummarizer(client=client, model="unknown-model")

    result = service.summarize("text")

    assert result.metrics.estimated_cost_usd == 0.0


def test_versioned_model_name_has_non_zero_cost():
    response = SimpleNamespace(
        output_text="summary",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=500, total_tokens=1500),
    )
    client = FakeOpenAIClient(response=response)
    service = OpenAISummarizer(client=client, model="gpt-4o-mini-2024-07-18")

    result = service.summarize("text")

    assert result.metrics.estimated_cost_usd > 0.0


class FakeEvaluator:
    def __init__(self, score: float):
        self.score = score

    def evaluate(self, text: str, summary: str) -> float:
        return self.score


def test_evaluator_score_is_sent_to_telemetry():
    response = SimpleNamespace(
        output_text="summary",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    client = FakeOpenAIClient(response=response)
    telemetry = SpyTelemetry()
    evaluator = FakeEvaluator(score=0.87)
    service = OpenAISummarizer(
        client=client,
        model="gpt-4o-mini",
        evaluator=evaluator,
        telemetry=telemetry,
    )

    service.summarize("text")

    assert telemetry.finished["evaluator_score"] == 0.87
