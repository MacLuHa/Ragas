
## Live результаты

- JSON-отчёт: [artifacts/ragas/live/results.json](artifacts/ragas/live/results.json)
- HTML-отчёт: [artifacts/ragas/live/results.html](artifacts/ragas/live/results.html)

Используемые метрики:

- `Faithfulness` — насколько ответ опирается на факты из контекста и не добавляет галлюцинации.
- `Answer Relevance` — насколько ответ релевантен исходному запросу/задаче.
- `Context Recall` — насколько полно ответ покрывает ключевые факты, которые есть в контексте.

Значения из `artifacts/ragas/live/results.json`:

```json
{
  "mode": "live",
  "candidate_source": "summarizer",
  "summarization_model": "gpt-4.1-mini",
  "aggregate_scores": {
    "faithfulness": 0.5794,
    "answer_relevance": 0.7182,
    "context_recall": 0.9667
  },
  "thresholds": {
    "faithfulness": 0.7,
    "answer_relevance": 0.65,
    "context_recall": 0.7
  },
  "threshold_checks": {
    "faithfulness": false,
    "answer_relevance": true,
    "context_recall": true
  },
  "passed": false
}
```

## Выводы

- Пайплайн live работает и оценивает **реальные ответы суммаризатора** (`candidate_source = summarizer`).
- `Answer Relevance` и `Context Recall` проходят пороги.
- Основная проблема качества сейчас — `Faithfulness` (`0.5794 < 0.7`), из-за чего общий quality gate не пройден (`passed = false`).
- Приоритет улучшений: уменьшать галлюцинации и усиливать сохранение фактов из исходного текста.
