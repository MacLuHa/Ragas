from __future__ import annotations

import os
from typing import Any, Optional


class OpenAIEvaluator:
    def __init__(
        self,
        client: Any,
        model: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model = model if model else os.getenv("OPENAI_MODEL_EVALUATOR", None)
        
    @staticmethod
    def create_evaluation_prompt(text: str, summary: str) -> str:
        return f"""
                Evaluate if this summary preserves the key points.

                Text:
                {text}

                Summary:
                {summary}

                Score from 0 to 1. Not add reduntant text, just return 0 or 1.
                """

    def evaluate(self, text: str, summary: str) -> float:
        evaluation_prompt = self.create_evaluation_prompt(text, summary)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise assistant."
                    ),
                },
                {"role": "user", "content": evaluation_prompt},
            ],
        )
        output_text = getattr(response, "output_text", "") or ""
        return float(output_text.strip())

