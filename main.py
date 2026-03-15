from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

# Allow running `python main.py` without manual PYTHONPATH setup.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarizer.service import OpenAISummarizer
from summarizer.telemetry import LangfuseTelemetrySink, NoopTelemetrySink
from summarizer.evaluator import OpenAIEvaluator


def build_httpx_client_from_env() -> httpx.Client | None:
    proxy_url = (
        os.getenv("PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
    )
    if not proxy_url:
        return None
    return httpx.Client(proxy=proxy_url, timeout=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize text with OpenAI and send metrics to Langfuse"
    )
    parser.add_argument("--text", help="Source text to summarize")
    parser.add_argument("--file", help="Path to text file to summarize")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Disable Langfuse telemetry",
    )
    return parser.parse_args()


def read_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Pass --text or --file")


def build_telemetry(disabled: bool):
    if disabled:
        return NoopTelemetrySink()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set. "
            "Fallback to no-op telemetry.",
            file=sys.stderr,
        )
        return NoopTelemetrySink()

    return LangfuseTelemetrySink(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        httpx_client=build_httpx_client_from_env(),
    )


def main() -> None:
    args = parse_args()
    source_text = read_input_text(args)
    openai_http_client = build_httpx_client_from_env()
    openai_client = OpenAI(http_client=openai_http_client)


    summarizer = OpenAISummarizer(
        client=openai_client,
        model=args.model,
        telemetry=build_telemetry(disabled=args.no_langfuse),
    )
    result = summarizer.summarize(source_text)

    print("=== Summary ===")
    print(result.summary)
    print("\n=== Metrics ===")
    print(json.dumps(result.metrics.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
