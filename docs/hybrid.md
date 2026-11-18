# Гибридный ассистент: малая модель + скиллы + большая модель

Архитектура от 2026-11-18. Решение — [ADR 0009](adr/0009-hybrid-router.md).

## Идея

Один вход, три источника ответа. Выбор — по правилам, а не по вкусу:

```
запрос → скиллы (точный ответ из каталога)      ← детерминированно, 0 ₽
       → малая троичная модель (наш голос)       ← дёшево, локально
       → большая модель по API (сложное)         ← дорого, редко
```

Скиллы впереди модели: если вопрос про цену тарифа, ответ не должен зависеть
от фантазии модели. Малая модель — для того, что она умеет: приветствия, тон,
пересказ контекста. Большая — когда нужен общие знания, код, рассуждение.

## Контракт модулей

Пакет `eva/`, только стандартная библиотека Python.

### `eva/types.py`

```python
class Message: role: str; content: str
class Request: messages: list[Message]; meta: dict          # meta: {"user_id", "channel"}
class Route: kind: str; reason: str; confidence: float      # kind: "skill"|"small"|"big"
class Answer: text: str; route: Route; skill: str|None; usage: dict;
              cost_rub: float; latency_ms: int
```

### `eva/catalog.py`

```python
load(path) -> Catalog
Catalog.get(name: str) -> dict|None          # услуга/тариф по имени или синониму
Catalog.all() -> list[dict]
Catalog.search(text: str) -> list[dict]      # грубый поиск по словам
```

Формат `data/catalog.json`: список `{"kind": "tariff"|"service", "name", "synonyms", "price_rub", "period", "limits", "notes", "sources": ["[1]"]}`.

### `eva/skills/`

```python
class Skill:                      # протокол
    name: str
    def match(self, text: str) -> bool
    def run(self, text: str, ctx: dict) -> SkillResult | None   # {"text", "sources", "catalog_ids"}

try_skills(text: str, ctx: dict) -> SkillResult | None          # registry.try_skills
```

Скиллы v1: `pricing` (цена тарифа/услуги), `hours` (часы работы), `refunds`
(возврат), `contacts` (почта, телефон, адрес). Скилл **не зовёт модель**.

### `eva/clients/openai_compat.py`

```python
class OpenAIClient:
    def __init__(self, base: str, key: str, model: str, timeout: float = 60, retries: int = 2)
    def chat(self, messages: list[Message], max_tokens: int = 512) -> ClientResult
    # ClientResult: {"text", "usage": {"prompt_tokens","completion_tokens","cost_rub"}, "raw": dict}
```

Ошибки: таймаут, 5xx, 429 — повтор с backoff; после исчерпания — `ClientError`.

### `eva/classifier.py`

```python
classify(text: str, cfg) -> Route
```

Правила v1 (без модели): длина, наличие маркеров рассуждения («почему»,
«объясни», «сравни», «напиши код», «переведи», «посчитай»), доля латиницы,
вопрос про факт из каталога. Конфиг порогов — в `config.py`.

### `eva/router.py` + `eva/policy.py`

```python
decide(req: Request, cfg) -> Route      # skills → classifier → small/big
answer(req: Request, cfg, deps) -> Answer
```

`policy`: порядок фолбэков (small→big при ошибке; big→small при 429), бюджет
на запрос (`max_cost_rub`), таймауты. Все решения и причины — в `Answer.route.reason`.

### `eva/server.py`

HTTP на stdlib: `POST /v1/chat/completions` — совместим с OpenAI-клиентами,
в ответе дополнительное поле `"eva": {"route": ..., "skill": ..., "cost_rub": ...}`.
`GET /health`. Без внешних зависимостей.

### `eva/telemetry.py`

Каждый ответ — строка в `logs/eva.jsonl`: время, маршрут, скилл, токены, рубли,
латентность. `usage_report(path)` → агрегат по маршрутам и дням.

### `eva/cli.py`

`python3 -m eva.cli ask "..." --explain` — ответ + маршрут + цена.
`python3 -m eva.cli serve --port 8000`.

## Правила

1. Скилл никогда не вызывает модель. Если скилл не нашёл совпадения — молчит.
2. Классификатор не имеет права ходить в сеть.
3. Провайдер больших моделей настраивается через `.env`, не хардкодом.
4. Каждый ответ несёт маршрут и цену — иначе не посчитать себестоимость.
5. Без сети всё работает: клиенты подменяются заглушкой в тестах.
