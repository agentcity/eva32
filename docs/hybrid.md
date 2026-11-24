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

## Метрики под чатом (задача 0145)

Стенд сравнения (`tools/chat_server.py`) показывает под чатом панель: две
колонки метрик по выбранным моделям и строка системы. Панель обновляется после
каждого ответа, внешних ресурсов не грузит, слушает только `127.0.0.1`.

### Откуда числа

| Что | Источник |
| --- | --- |
| размер файла модели | `os.stat().st_size` того же `.vtm` |
| параметры, KV-кэш, словарь, слои, контекст | таблица тензоров и JSON-метаданные `.vtm` (читаются без загрузки модели) |
| тип хранения (trit/int8/fp16) | `kind` тензоров + заголовки сегментов TRT1/INT8 + `quantization.scale_dtype` из меты |
| источник (демо / холодный экспорт / QAT) | `base_model`, `checkpoint`, `step`, `kv_cache` из меты; без меты — эвристика «vocab 128 = демо» |
| бит на параметр | байты весов в файле × 8 / число параметров |
| RSS до/после загрузки | `/proc/self/status` (Linux) или `proc_pidinfo` (macOS) вокруг вызова фабрики бэкенда; пик — `getrusage().ru_maxrss` |
| TTFT | время от старта запроса до первого куска текста |
| prefill | то же время до первого токена (prefill + один шаг декода — рантайм не разделяет их отдельным событием) |
| декод | `elapsed_ms − first_token_ms` |
| ток/с декода | completion-токены / время декода |
| CPU запроса | `time.process_time()` до и после всего запроса (процесс, не поток: две колонки идут параллельно) |
| `VTM_THREADS` | переменная окружения; если не задана — «по умолчанию» (min(ядер, 8)) |
| load average | `os.getloadavg()` |
| свободная RAM | `/proc/meminfo` (`MemAvailable`) или `vm_stat` (free+inactive+speculative) |
| свободный диск | `shutil.disk_usage(корень репозитория)` |
| Ollama `size`/`size_vram` | `GET http://127.0.0.1:11434/api/ps` — только для локальной Ollama; в панели это строка «резидентная копия» |

Чего снять нельзя, того панель не выдумывает: API отдаёт `null`, в UI — «—».
Пример: у Ollama нет ни размера файла на диске, ни числа параметров, ни
KV-кэша в нашем смысле; у заглушек в тестах нет ни загрузки, ни холодного
старта.

### Эндпоинты

- `GET /api/models` — карточка на каждую модель реестра: `disk_bytes`,
  `params`, `vocab`, `layers`, `context`, `kv_bytes`, `kv_in_file`, `storage`,
  `source`, `bits_per_param`, `present`, `loaded`, `load` (RSS до/после,
  дельта, пик, время загрузки), `cache_prefix`, `last` (метрики последнего
  ответа). Абсолютные пути наружу не отдаются: только id модели из реестра.
- `GET /api/system` — `load`, `cpu_count`, `ram` (total/free/source), `disk`,
  `process` (pid, rss, peak, source), `vtm_threads` (runtime/config), `platform`.
- `POST /api/chat` — прежние события `start`/`delta`/`done`/`error`; в `start`
  добавлено `cold` (модель грузилась этим запросом), в `done` — `prefill_ms`,
  `decode_ms`, `decode_tokens_per_second`, `finish_reason` (`stop`/`eos`/
  `length`), `first_token` («холодный»/«тёплый»), `cache_prefix`, `load`
  (RSS-дельта загрузки), `ollama` (из `/api/ps`); событие `end` несёт
  `metrics`: `wall_ms`, `cpu_ms`, `cpu_cores`, снимок системы и `start_memory`.

### Тонкости, которые важно знать

1. **prefill = TTFT.** Рантайм ВТМ (`vtm_api_generate`) не сообщает отдельно
   момент конца prefill; первый токен появляется после prefill и одного шага
   декода, поэтому панель называет это время prefill и честно показывает его
   же как TTFT, а не выдумывает разницу.
2. **Кэш префикса ВТМ — оценка.** C-рантайм не отдаёт счётчик попаданий
   (`vtm_api_*` его не имеет), поэтому сервер сравнивает текущий промпт с
   прошлым: попадание возможно, когда прошлый промпт целиком входит в текущий
   (кэш переиспользует всё, кроме последнего токена). В ответе стоит
   `"estimated": true`; в UI — «(оценка)». Если кэш выключен — «выключен», если
   настройка ещё неизвестна (модель не загружена) — «—».
3. **RSS-дельта честная только для ВТМ.** Модель ВТМ живёт в процессе
   `chat_server` через `ctypes`, поэтому дельта RSS «до/после загрузки» — это
   её память (вместе с токенизатором и загрузчиком). Для Ollama процесс чужой:
   `proc_pidinfo` его не видит, поэтому размер берётся из `/api/ps`, а RSS
   процесса Ollama — «—».
4. **Дельта RSS может быть 0 или отрицательной.** Если аллокатор
   переиспользовал уже резидентные страницы (или во время загрузки что-то
   освободилось), RSS не вырастет, хотя модель загрузилась. Это не ошибка
   измерения: `rss_before`/`rss_after` — сырые числа, пик процесса и
   `loaded: true` показывают, что загрузка была.
5. **CPU-время — на запрос, а не на колонку.** `time.process_time()` считает
   все потоки процесса; две колонки идут параллельно, поэтому `cpu_cores > 1`
   — норма (работали два ядра).
6. **Сеть.** Единственный сетевой вызов сервера — `GET /api/ps` у Ollama на
   `127.0.0.1:11434`, и только для реального адаптера Ollama. Тесты подставляют
   заглушки без `probe_ps`, поэтому идут без сети.

### Проверка

```bash
python3 tools/chat_server.py --port 8765
curl -s http://127.0.0.1:8765/api/models | python3 -m json.tool | head -40
curl -s http://127.0.0.1:8765/api/system | python3 -m json.tool
# размер в карточке == stat:
python3 - <<'PY'
import json, urllib.request
card = json.load(urllib.request.urlopen("http://127.0.0.1:8765/api/models"))["models"][0]
print(card["disk_bytes"] == __import__("os").path.getsize(card["model"]))
PY
```

Тесты: `python3 -m unittest tests.test_chat_server -v` (48 тестов, без сети и
без настоящих моделей).
