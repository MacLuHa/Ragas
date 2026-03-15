# OpenAI Summarization + Langfuse Cloud

Небольшой проект для суммаризации текста через OpenAI с отправкой метрик в Langfuse Cloud.

## Что собирается

Обязательные метрики:
- Время выполнения (`duration_ms`)
- Входные и выходные данные (текст запроса и summary в trace/generation)
- Использование токенов и стоимость (`input_tokens`, `output_tokens`, `total_tokens`, `estimated_cost_usd`)
- Длина исходного и итогового текста (`input_text_length`, `output_text_length`)

## Установка

```bash
uv sync
```

Или через `pip`:

```bash
pip install -e . pytest
```

Создай `.env` на основе `.env.example`.

Укажи:
- `LANGFUSE_HOST` (по умолчанию `https://cloud.langfuse.com`)
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `OPENAI_MODEL_SUMMARIZATION` (например, `gpt-4o-mini`)
- `OPENAI_MODEL_EVALUATOR` (если задан, его score отправляется в Langfuse как `custom_evaluator`)

Если нужен корпоративный прокси:
- `PROXY_URL` (например, `http://user:password@proxy.host:port`)

## Запуск

```bash
python main.py --text "Большой текст для суммаризации..."
```

или

```bash
python main.py --file ./examples/article.txt
```

Без Langfuse:

```bash
python main.py --text "Текст" --no-langfuse
```

## Тестовые примеры

```bash
pytest -q
```

Тесты проверяют:
- корректный сбор всех обязательных метрик;
- передачу входного/выходного текста в telemetry;
- расчет стоимости и поведение для неизвестной модели.
