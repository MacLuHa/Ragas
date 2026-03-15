from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from openai import OpenAI

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from summarizer.service import OpenAISummarizer
from summarizer.telemetry import NoopTelemetrySink


METRIC_KEYS = ("faithfulness", "answer_relevance", "context_recall")
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevance": 0.65,
    "context_recall": 0.70,
}
REQUIRED_GOLDEN_FIELDS = {
    "id",
    "input_text",
    "reference_summary",
    "key_facts",
    "contexts",
    "candidate_summary",
}


@dataclass(slots=True)
class ScoreRow:
    sample_id: str
    faithfulness: float
    answer_relevance: float
    context_recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_recall": round(self.context_recall, 4),
        }


class _ReplayResponsesAPI:
    def __init__(self, summaries_by_text: dict[str, str]) -> None:
        self._summaries_by_text = summaries_by_text

    def create(self, **kwargs: Any) -> Any:
        text = ""
        for item in kwargs.get("input", []):
            if item.get("role") == "user":
                text = item.get("content", "")
                break

        output_text = self._summaries_by_text.get(text, "")
        input_tokens = max(1, len(text) // 4)
        output_tokens = max(1, len(output_text) // 4)
        usage = type(
            "Usage",
            (),
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )()
        return type("Response", (), {"output_text": output_text, "usage": usage})()


class _ReplayOpenAIClient:
    def __init__(self, summaries_by_text: dict[str, str]) -> None:
        self.responses = _ReplayResponsesAPI(summaries_by_text)


def _build_httpx_client_from_env() -> httpx.Client | None:
    proxy_url = (
        os.getenv("PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
    )
    if not proxy_url:
        return None
    return httpx.Client(proxy=proxy_url, timeout=30.0)


def load_goldens(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("goldens file must contain a JSON list")
    return payload


def validate_goldens(goldens: list[dict[str, Any]]) -> None:
    if not goldens:
        raise ValueError("goldens list must not be empty")
    if not (10 <= len(goldens) <= 20):
        raise ValueError("goldens list size must be between 10 and 20 items")

    seen_ids: set[str] = set()
    for idx, sample in enumerate(goldens):
        missing = REQUIRED_GOLDEN_FIELDS - set(sample.keys())
        if missing:
            raise ValueError(f"golden[{idx}] is missing fields: {sorted(missing)}")

        sample_id = str(sample["id"]).strip()
        if not sample_id:
            raise ValueError(f"golden[{idx}] has empty id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate golden id: {sample_id}")
        seen_ids.add(sample_id)

        key_facts = sample["key_facts"]
        contexts = sample["contexts"]
        if not isinstance(key_facts, list) or not key_facts or not all(
            isinstance(item, str) and item.strip() for item in key_facts
        ):
            raise ValueError(f"golden[{sample_id}] has invalid key_facts")
        if not isinstance(contexts, list) or not contexts or not all(
            isinstance(item, str) and item.strip() for item in contexts
        ):
            raise ValueError(f"golden[{sample_id}] has invalid contexts")

        for field in ("input_text", "reference_summary", "candidate_summary"):
            value = sample[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"golden[{sample_id}] has invalid '{field}'")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9]+", _normalize_text(text)))


def _fact_hit_ratio(facts: list[str], text: str) -> float:
    normalized_text = _normalize_text(text)
    hit_count = sum(1 for fact in facts if _normalize_text(fact) in normalized_text)
    return hit_count / len(facts)


def _answer_relevance_score(question: str, answer: str, facts: list[str]) -> float:
    question_tokens = _tokenize(question)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0

    overlap = len(answer_tokens & question_tokens) / len(answer_tokens)
    fact_hit = _fact_hit_ratio(facts, answer)
    return min(1.0, 0.55 * overlap + 0.45 * fact_hit)


def _context_recall_score(contexts: list[str], facts: list[str], answer: str) -> float:
    context_blob = _normalize_text(" ".join(contexts))
    answer_blob = _normalize_text(answer)
    covered = 0
    for fact in facts:
        normalized_fact = _normalize_text(fact)
        if normalized_fact in context_blob and normalized_fact in answer_blob:
            covered += 1
    return covered / len(facts)


def _mock_score_sample(sample: dict[str, Any]) -> ScoreRow:
    facts = sample["key_facts"]
    answer = sample["candidate_summary"]
    question = sample["input_text"]
    contexts = sample["contexts"]

    faithfulness = _fact_hit_ratio(facts, answer)
    answer_relevance = _answer_relevance_score(question, answer, facts)
    context_recall = _context_recall_score(contexts, facts, answer)

    return ScoreRow(
        sample_id=str(sample["id"]),
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_recall=context_recall,
    )


def _aggregate(rows: list[ScoreRow]) -> dict[str, float]:
    return {
        "faithfulness": round(mean(row.faithfulness for row in rows), 4),
        "answer_relevance": round(mean(row.answer_relevance for row in rows), 4),
        "context_recall": round(mean(row.context_recall for row in rows), 4),
    }


def _evaluate_mock(goldens: list[dict[str, Any]]) -> tuple[list[ScoreRow], dict[str, float]]:
    rows = [_mock_score_sample(sample) for sample in goldens]
    return rows, _aggregate(rows)


def _resolve_metric_aliases(row: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "faithfulness": ("faithfulness",),
        "answer_relevance": ("answer_relevance", "answer_relevancy"),
        "context_recall": ("context_recall",),
    }
    out: dict[str, float] = {}
    for target, keys in aliases.items():
        value = 0.0
        for key in keys:
            if key in row and isinstance(row[key], (float, int)):
                value = float(row[key])
                break
        out[target] = round(value, 4)
    return out


def _evaluate_live(goldens: list[dict[str, Any]]) -> tuple[list[ScoreRow], dict[str, float]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Ragas live mode dependencies/imports are unavailable"
        ) from exc

    dataset_rows = [
        {
            "question": sample["input_text"],
            "answer": sample["candidate_summary"],
            "contexts": sample["contexts"],
            "ground_truth": sample["reference_summary"],
        }
        for sample in goldens
    ]
    llm_model = os.getenv("OPENAI_MODEL_EVALUATOR", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_http_client = _build_httpx_client_from_env()
    llm = ChatOpenAI(model=llm_model, temperature=0, http_client=openai_http_client)
    embeddings = LCOpenAIEmbeddings(model=embedding_model, http_client=openai_http_client)

    ragas_result = evaluate(
        dataset=Dataset.from_list(dataset_rows),
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    result_rows: list[dict[str, Any]]
    if hasattr(ragas_result, "to_pandas"):
        result_rows = ragas_result.to_pandas().to_dict(orient="records")
    elif isinstance(ragas_result, dict):
        result_rows = [ragas_result]
    else:  # pragma: no cover
        raise RuntimeError("Unsupported ragas result shape")

    per_sample: list[ScoreRow] = []
    for sample, ragas_row in zip(goldens, result_rows):
        resolved = _resolve_metric_aliases(ragas_row)
        per_sample.append(
            ScoreRow(
                sample_id=str(sample["id"]),
                faithfulness=resolved["faithfulness"],
                answer_relevance=resolved["answer_relevance"],
                context_recall=resolved["context_recall"],
            )
        )
    return per_sample, _aggregate(per_sample)


def _evaluate(
    goldens: list[dict[str, Any]],
    mode: str,
) -> tuple[list[ScoreRow], dict[str, float]]:
    if mode == "mock":
        return _evaluate_mock(goldens)
    if mode == "live":
        return _evaluate_live(goldens)
    raise ValueError(f"Unsupported mode: {mode}")


def _generate_candidate_summaries(
    goldens: list[dict[str, Any]],
    mode: str,
    candidate_source: str,
    summarization_model: str,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    if candidate_source == "golden":
        return goldens

    if mode == "mock":
        replay_map = {sample["input_text"]: sample["candidate_summary"] for sample in goldens}
        client: Any = _ReplayOpenAIClient(replay_map)
    else:
        client = OpenAI(http_client=_build_httpx_client_from_env())

    summarizer = OpenAISummarizer(
        client=client,
        model=summarization_model,
        max_output_tokens=max_output_tokens,
        telemetry=NoopTelemetrySink(),
    )

    generated_rows: list[dict[str, Any]] = []
    for sample in goldens:
        summary = summarizer.summarize(sample["input_text"]).summary
        updated = dict(sample)
        updated["candidate_summary"] = summary
        generated_rows.append(updated)
    return generated_rows


def _evaluate_thresholds(
    aggregate: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, dict[str, bool]]:
    checks = {metric: aggregate[metric] >= thresholds[metric] for metric in METRIC_KEYS}
    return all(checks.values()), checks


def write_results_json(report: dict[str, Any], output_dir: str | Path, mode: str) -> Path:
    output_path = Path(output_dir) / mode / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return output_path


def write_results_html(report: dict[str, Any], output_dir: str | Path, mode: str) -> Path:
    output_path = Path(output_dir) / mode / "results.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate_rows = "".join(
        f"<tr><td>{name}</td><td>{report['aggregate_scores'][name]:.4f}</td>"
        f"<td>{report['thresholds'][name]:.4f}</td>"
        f"<td>{'PASS' if report['threshold_checks'][name] else 'FAIL'}</td></tr>"
        for name in METRIC_KEYS
    )
    sample_rows = "".join(
        "<tr>"
        f"<td>{item['id']}</td>"
        f"<td>{item['faithfulness']:.4f}</td>"
        f"<td>{item['answer_relevance']:.4f}</td>"
        f"<td>{item['context_recall']:.4f}</td>"
        "</tr>"
        for item in report["per_sample_scores"]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ragas Evaluation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f6f6f6; }}
  </style>
</head>
<body>
  <h1>Ragas Evaluation Report</h1>
  <p><strong>Mode:</strong> {report['mode']}</p>
  <p><strong>Candidate source:</strong> {report['candidate_source']}</p>
  <p><strong>Summarization model:</strong> {report['summarization_model']}</p>
  <p><strong>Timestamp:</strong> {report['timestamp']}</p>
  <p><strong>Status:</strong> {"PASS" if report['passed'] else "FAIL"}</p>
  <h2>Aggregate Scores</h2>
  <table>
    <thead><tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Gate</th></tr></thead>
    <tbody>{aggregate_rows}</tbody>
  </table>
  <h2>Per Sample Scores</h2>
  <table>
    <thead><tr><th>ID</th><th>Faithfulness</th><th>Answer Relevance</th><th>Context Recall</th></tr></thead>
    <tbody>{sample_rows}</tbody>
  </table>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)
    return output_path


def run_evaluation(
    goldens_path: str | Path,
    output_dir: str | Path = "artifacts/ragas",
    mode: str = "mock",
    thresholds: dict[str, float] | None = None,
    candidate_source: str = "summarizer",
    summarization_model: str | None = None,
    max_output_tokens: int = 500,
) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
    summarization_model = summarization_model or os.getenv("OPENAI_MODEL_SUMMARIZATION", "gpt-4o-mini")

    goldens = load_goldens(goldens_path)
    validate_goldens(goldens)
    goldens_with_candidates = _generate_candidate_summaries(
        goldens=goldens,
        mode=mode,
        candidate_source=candidate_source,
        summarization_model=summarization_model,
        max_output_tokens=max_output_tokens,
    )

    per_sample, aggregate = _evaluate(goldens_with_candidates, mode)
    passed, threshold_checks = _evaluate_thresholds(aggregate, thresholds)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "candidate_source": candidate_source,
        "summarization_model": summarization_model,
        "aggregate_scores": aggregate,
        "per_sample_scores": [row.to_dict() for row in per_sample],
        "thresholds": thresholds,
        "threshold_checks": threshold_checks,
        "passed": passed,
    }

    write_results_json(report, output_dir, mode)
    write_results_html(report, output_dir, mode)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas quality evaluation")
    parser.add_argument("--goldens", default="tests/goldens.json", help="Path to goldens dataset")
    parser.add_argument("--output", default="artifacts/ragas", help="Output dir for reports")
    parser.add_argument("--mode", choices=["mock", "live"], default="live", help="Evaluation mode")
    parser.add_argument(
        "--candidate-source",
        choices=["summarizer", "golden"],
        default="summarizer",
        help="Source of candidate summaries",
    )
    parser.add_argument("--summarization-model", default=None, help="Model for OpenAISummarizer")
    parser.add_argument("--max-output-tokens", type=int, default=500, help="Max output tokens for summarizer")
    parser.add_argument("--faithfulness-threshold", type=float, default=DEFAULT_THRESHOLDS["faithfulness"])
    parser.add_argument("--answer-relevance-threshold", type=float, default=DEFAULT_THRESHOLDS["answer_relevance"])
    parser.add_argument("--context-recall-threshold", type=float, default=DEFAULT_THRESHOLDS["context_recall"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {
        "faithfulness": args.faithfulness_threshold,
        "answer_relevance": args.answer_relevance_threshold,
        "context_recall": args.context_recall_threshold,
    }
    report = run_evaluation(
        goldens_path=args.goldens,
        output_dir=args.output,
        mode=args.mode,
        thresholds=thresholds,
        candidate_source=args.candidate_source,
        summarization_model=args.summarization_model,
        max_output_tokens=args.max_output_tokens,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
