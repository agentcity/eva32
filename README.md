# EVA32

> Актуально на 2026-11-24.

Бандл модели для чата orioncode: веса (ссылка + ревизия), промпт, конфиг генерации, сервинг, eval.

Чего здесь нет и не должно быть: кода чата и Symfony (orioncode), RAG-индекса и документов (pgvector),
секретов, весов в git.

## Что сейчас (2026-11-24)

- **ВТМ считает 3B Q4** — 2,245 ГиБ (Q4_1) / 1,92 ГиБ (Q4_0), импорт из ollama GGUF
  Q4_K_M в kind 5 ([0155](ai/tasks/0155-stream-q4-writer.md),
  [0160](ai/tasks/0160-q4-asym-import.md)); **7B Q4** — 3,831 ГиБ = 95,79% предела
  4 ГиБ ([0156](ai/tasks/0156-q4-7b.md), [ADR 0016](docs/adr/0016-vtm-4gib-limit.md)).
  Скорость: 3B — 7–8 ток/с в сессии, 7B — 4,5 ток/с тёплым
  ([0162](ai/tasks/0162-q4-prefill-batch.md), [0161](ai/tasks/0161-q4-7b-slow.md)).
- **Данные — 75 000 пар** для QAT: код 45,4%, бизнес 35,0%, общее 17,4%, поведение 2,2%
  (`runs/mixture/stats.json`); сам QAT ждёт GPU ([0008](ai/tasks/0008-qat-pilot-run.md)).
- **Эталон:** qwen2.5:3b — 65/72 = 90,3% после исправления чекера
  ([0140](ai/tasks/0140-checker-morphology.md)); наша 3B Q4_1 с компактным промптом —
  20 вопросов: reasoning 8/8, marketing 8/8, clarify 2/8
  ([0139](ai/tasks/0139-eval-runner.md)).
- **2305** Python-тестов, CI зелёный на ubuntu и macos. Полная сводка —
  [docs/status.md](docs/status.md), индекс документации с датами —
  [docs/README.md](docs/README.md).

## Определяющий результат: триты — не сжатие, а обучение

Мы проверили, можно ли получить рабочую троичную модель из готовой Qwen **без обучения**.
Проверили на 0,5B и 3B, четырьмя схемами. Ответ — нет:

| Схема | 0,5B / 3B, во сколько раз хуже bf16 | Читаемо |
|---|---|---|
| Холодные триты (Linear) | ×3,9e5 / ×4,5e6 | нет |
| + тритовая таблица словаря | ×2,8e6 / ×1,7e7 | нет |
| Два трита на вес (2 бита) | ×217, декод вырождается | нет |
| **int8** | **×1,004** | да |
| **Q4 (наш формат, kind 5)** | паритет C↔Python 1,4e-6…2,1e-6; NEON Q4 7,8× | да |
| **Триты после QAT** | — | цель |

Три следствия, которые задают путь:

1. **Урон растёт с размером модели**, а не падает: 0,5B → 3B дало ×3,9e5 → ×4,5e6.
2. **Главный источник урона — Linear, а не таблица словаря** (изоляция на 0,5B:
   таблица — ×4043, Linear — от ×6,3e5 при любой таблице).
3. **Два трита не окупаются:** стоят 3,2–3,45 бита, а не 2, и проигрывают int8.

Отсюда продуктовая картина для 3B: **Q4 1,92 ГиБ** (Q4_0, живой стенд сейчас;
Q4_1 — 2,245 ГиБ), **int8 ≈ 3,1 ГБ** (эталон качества), **QAT-триты ≈ 0,69 ГБ**
(цель, только после обучения). Триты не сжимают готовую модель — они результат
обучения под точность.

Подробно: [docs/quantization-verdict.md](docs/quantization-verdict.md).

## С чего начать

- **[docs/README.md](docs/README.md) — индекс документации**: одна строка на документ и
  «актуально на»; свежесть проверяет `python3 tools/check_docs_dates.py`.
- [docs/status.md](docs/status.md) — что работает, что открыто, риски; у каждого числа — задача или команда.
- [docs/quantization-verdict.md](docs/quantization-verdict.md) — определяющий результат по тритам и Q4.
- [docs/model-zoo.md](docs/model-zoo.md) — какие модели лежат и что из них живое.

Дальше по интересам:

- [docs/architecture.md](docs/architecture.md) — устройство системы; [docs/roadmap.md](docs/roadmap.md) — что осталось.
- [docs/vtm.md](docs/vtm.md) — спецификация ВТМ (контейнер, опкоды, C-ABI); [docs/vtm-perf.md](docs/vtm-perf.md) — числа.
- [docs/hybrid.md](docs/hybrid.md) — контракт гибрида; [docs/roadmap.md](docs/roadmap.md) — фазы; [docs/adr/](docs/adr/) — решения.
- [docs/contract.md](docs/contract.md) — контракт с orioncode: промпт, контекст, стриминг, ошибки.
- [docs/work-plan.md](docs/work-plan.md) — очерёдность работ, гейты; [docs/plan-russian.md](docs/plan-russian.md) — русский язык.
- [docs/quantization.md](docs/quantization.md) — план и история гипотез (итог — verdict).
- [docs/trit-format.md](docs/trit-format.md) — 20 тритов в 32-битном слове; [docs/trit-emulation.md](docs/trit-emulation.md) — исполнение.
- [docs/distillation.md](docs/distillation.md) — дистилляция; [docs/data-plan.md](docs/data-plan.md) — данные; [docs/qat.md](docs/qat.md) — QAT.
- [docs/golden.md](docs/golden.md) — golden: 72 компетенции и 45 каталога.
- [docs/gpu-checklist.md](docs/gpu-checklist.md) — подъём на GPU; [docs/rental-runbook.md](docs/rental-runbook.md) — аренда.
- `MODEL_CARD.md` — что за модель, лицензия, ограничения.
- `bench/ternary_bench.c` — замер int8 / int4 / тритов; `tools/trit_codec.py` — кодек и контрольные векторы.
- `prompts/system.ru.md` — персона и правила (в чате по умолчанию `prompts/system.compact.md`).
- `config/generation.yaml` — параметры генерации; `serving/compose.yaml` — vLLM на GPU-сервере.
- `eval/` — золотой набор и прогон; `dev/` — локальные эксперименты (Ollama), не прод.
- `Makefile` — команды: стенд, eval, статус, vLLM.

## Команды

```bash
make                 # список команд
make status          # что запущено: процессы, порты, API, что в памяти
make bundle          # хеш бандла (промпт + конфиг) — версия для eval/CHANGELOG
make dev             # поднять стенд и прогнать золотой набор
make eval-dev        # только прогон на стенде (Ollama, :11434)
make eval-dev REPEAT=3   # каждая строка трижды — видно, что нестабильно
make dev-stop        # выгрузить модель и погасить сервер
make eval            # прогон против vLLM (:8000)
make compose-up      # поднять vLLM на GPU-сервере (нужен .env)

python3 eval/run_eval.py --dry-run   # тела запросов без отправки — сверка с docs/contract.md

python3 tools/chat.py --backend vtm --model runs/q4/3b-q4_1.vtm   # REPL: /stats, /reset
python3 tools/chat.py --backend ollama --model qwen2.5:3b         # та же оболочка для Ollama
python3 tools/chat_server.py                                      # 127.0.0.1:8765, два ответа рядом
```

Чат — один интерфейс для трёх бэкендов (`vtm|ollama|openai`): история, стриминг,
`/stats` с TTFT и ток/с, стоимость по API. Веб-стенд показывает ответы ВТМ и Ollama
рядом и панель метрик — диск, RAM, CPU, prefill/декод
([0137](ai/tasks/0137-chat-cli.md), [0138](ai/tasks/0138-chat-web.md),
[0145](ai/tasks/0145-chat-stats-panel.md)).

Результаты прогонов — `eval/results/eval-*.json`: вердикты по строкам, ответы модели и `bundle`.

## Гибрид

Ассистент — не только веса: скиллы отвечают из каталога (0 ₽, без галлюцинаций), малая
модель — тон и простое, большая — то, где малой не хватает знаний. Маршрут выбирается
правилами, у каждого ответа есть цена.

- `docs/hybrid.md` — контракт модулей: типы, маршрутизатор, сервер, телеметрия.
- `docs/adr/0009-hybrid-router.md` — почему гибрид, а не своя большая модель.
- `docs/hybrid-runbook.md` — запуск, `.env`, подключение из orioncode, метрики и 429.

