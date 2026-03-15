from __future__ import annotations

import json
from pathlib import Path

from summarizer.ragas_pipeline import (
    DEFAULT_THRESHOLDS,
    load_goldens,
    run_evaluation,
    validate_goldens,
)


GOLDENS_PATH = Path("tests/goldens.json")


def test_goldens_schema_and_size() -> None:
    goldens = load_goldens(GOLDENS_PATH)
    validate_goldens(goldens)

    assert len(goldens) == 15
    assert len({item["id"] for item in goldens}) == 15
    assert all(item["key_facts"] for item in goldens)


def test_ragas_pipeline_mock_writes_reports_and_passes_thresholds(tmp_path: Path) -> None:
    report = run_evaluation(
        goldens_path=GOLDENS_PATH,
        output_dir=tmp_path,
        mode="mock",
        thresholds=DEFAULT_THRESHOLDS,
    )

    json_path = tmp_path / "mock" / "results.json"
    html_path = tmp_path / "mock" / "results.html"

    assert json_path.exists()
    assert html_path.exists()
    assert report["mode"] == "mock"
    assert report["candidate_source"] == "summarizer"
    assert report["passed"] is True
    assert len(report["per_sample_scores"]) == 15

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["aggregate_scores"]["faithfulness"] >= DEFAULT_THRESHOLDS["faithfulness"]
    assert payload["aggregate_scores"]["answer_relevance"] >= DEFAULT_THRESHOLDS["answer_relevance"]
    assert payload["aggregate_scores"]["context_recall"] >= DEFAULT_THRESHOLDS["context_recall"]


def test_ragas_pipeline_fails_with_strict_thresholds(tmp_path: Path) -> None:
    strict_thresholds = {
        "faithfulness": 1.01,
        "answer_relevance": 1.01,
        "context_recall": 1.01,
    }
    report = run_evaluation(
        goldens_path=GOLDENS_PATH,
        output_dir=tmp_path,
        mode="mock",
        thresholds=strict_thresholds,
    )

    assert report["passed"] is False
    assert report["threshold_checks"]["faithfulness"] is False
    assert report["threshold_checks"]["answer_relevance"] is False
    assert report["threshold_checks"]["context_recall"] is False
