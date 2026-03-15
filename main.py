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
from summarizer.telemetry import NoopTelemetrySink


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
        description="Summarize text with OpenAI and collect local metrics"
    )
    parser.add_argument("--text", help="Source text to summarize")
    parser.add_argument("--file", help="Path to text file to summarize")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    return parser.parse_args()


def read_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Pass --text or --file")


def main() -> None:
    args = parse_args()
    source_text = read_input_text(args)
    openai_http_client = build_httpx_client_from_env()
    openai_client = OpenAI(http_client=openai_http_client)


    summarizer = OpenAISummarizer(
        client=openai_client,
        model=args.model,
        telemetry=NoopTelemetrySink(),
    )
    result = summarizer.summarize(source_text)

    print("=== Summary ===")
    print(result.summary)
    print("\n=== Metrics ===")
    print(json.dumps(result.metrics.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
