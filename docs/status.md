# Состояние проекта EVA32

> Актуально на 2026-11-26.

Снимок на **2026-11-26**. У каждого числа — задача или команда, которой оно
получено; «не измерено» значит ровно это. Реестр задач —
[`ai/tasks/index.json`](../ai/tasks/index.json): **201 задача**, из них 183 `done`,
8 `pending`, 8 `in_progress`, 2 `blocked` (`python3 tools/check_tasks.py`).

## Что сейчас

- **3B Q4_K считается на GPU, а не только на CPU.** Metal (режим `auto`) —
  **24–30 ток/с**, CPU simd — **16–20**; гибрид (`hybrid`) оставляет декод на
  CPU, а prefill отдаёт GPU ([0204](../ai/tasks/0204-metal-phase3.md),
  [0198](../ai/tasks/0198-hybrid-gpu-prefill.md),
  [ADR 0021](adr/0021-metal-gpu-backend.md)).
- **Prefill перестал быть узким местом диалога.** TTFT свежего диалога —
  **0,34–0,35 с** (прогретая модель, кэш системного промпта), на чистом CPU —
  **1,6–1,9 с**; GPU-prefill **1,7–2,4×** быстрее CPU
  ([0198](../ai/tasks/0198-hybrid-gpu-prefill.md),
  [0204](../ai/tasks/0204-metal-phase3.md)).
- **30B MoE работает на 16 ГиБ, но медленно.** Qwen3-30B-A3B — **17,28 ГиБ**,
  **1,88 ток/с**, TTFT **13,4 с**: эксперты читаются `pread` в кэш 1792 МБ,
  6 I/O-потоков параллельно со счётом ([0175](../ai/tasks/0175-moe-pread-cache.md),
  [0177](../ai/tasks/0177-moe-speed.md), [0180](../ai/tasks/0180-cache-plus-batch.md),
  [moe.md](moe.md)). Упор — в диск, а не в ядра.
- **Голос — Silero.** `baya`/`kseniya`/`xenia`: **0,04–0,18 с** на фразу после
  prewarm, worker **0,84–0,93 ГБ** ([0193](../ai/tasks/0193-silero-tts.md),
  [0195](../ai/tasks/0195-silero-worker.md)); ElevenLabs — по ключу,
  `say` — аварийно; прежние движки удалены
  ([0197](../ai/tasks/0197-switch-to-silero-only.md)).
- **Клиент — EVA.app, а не браузер.** ≈0,7 МБ: ⌥Space, wake-word «Привет, Ева»,
  follow-up, круг состояний, компактный режим, «Стоп, Ева»
  ([0184](../ai/tasks/0184-native-voice-app.md),
  [0186](../ai/tasks/0186-eva-circle-visual.md),
  [0187](../ai/tasks/0187-wake-word.md),
  [0188](../ai/tasks/0188-compact-floating-circle.md),
  [0190](../ai/tasks/0190-followup-listening.md),
  [0194](../ai/tasks/0194-stop-command.md);
  [ADR 0020](adr/0020-clients-and-cloud.md)).
- **Qwen-шаблон и бюджет размышлений.** ChatML применяется автоматически,
  `<think>` по умолчанию выключен, `--reasoning-budget N` ограничивает
  рассуждение и досказывает ответ ([0200](../ai/tasks/0200-qwen-template-thinking.md)).
- **Речь нормализуется до синтеза:** числа, проценты, даты, телефоны, email,
  знаки и сокращения ([0205](../ai/tasks/0205-tts-numbers.md),
  [0206](../ai/tasks/0206-tts-symbols.md)); о себе Ева говорит в женском роде
  ([0207](../ai/tasks/0207-feminine-speech.md)).
- **Плагины — в работе** ([0209](../ai/tasks/0209-plugin-runtime.md),
  [ADR 0022](adr/0022-plugins.md)): контракт `plugin.json` + способности
  `skill.fetch`/`filter.run`/`tts.engine`/`command`, но рабочих плагинов и
  включения пока нет. Это статус, а не готовый API.
- **Данные — 75 000 пар** для QAT: код 45,4%, бизнес 35,0%, общее 17,4%,
  поведение 2,2% (`runs/mixture/stats.json`); QAT ждёт GPU
  ([0008](../ai/tasks/0008-qat-pilot-run.md)).
- **Эталон:** qwen2.5:3b — **65/72 = 90,3%** после исправления чекера
  ([0140](../ai/tasks/0140-checker-morphology.md)). Наша 3B Q4_1 с компактным
  промптом: reasoning **8/8**, marketing **8/8**, clarify **2/8**
  ([0139](../ai/tasks/0139-eval-runner.md)).

## Числа (2026-11-26)

| Что измерено | Число | Чем проверено | Задача |
|---|---:|---|---|
| 3B Q4_K, декод Metal (`auto`) | **24–30 ток/с** | `bench_backend --metal`, `tools/chat.py --stats` | [0204](../ai/tasks/0204-metal-phase3.md) |
| 3B Q4_K, декод CPU simd | **16–20 ток/с** | `bench_backend --metal`, парный замер | [0204](../ai/tasks/0204-metal-phase3.md) |
| Prefill GPU против CPU | **1,7–2,4×** | `bench_backend --metal` | [0198](../ai/tasks/0198-hybrid-gpu-prefill.md), [0204](../ai/tasks/0204-metal-phase3.md) |
| TTFT свежего диалога (кэш промпта) | **0,34–0,35 с** | живой замер чата 2026-11-26, `--stats` | [0204](../ai/tasks/0204-metal-phase3.md) |
| TTFT 64 токена, чистый CPU | **1,64–1,67 с** | `bench_backend --metal`, парный замер | [0198](../ai/tasks/0198-hybrid-gpu-prefill.md) |
| 30B MoE, декод | **1,88 ток/с** | `vtm run --stats`, `VTM_MOE_CACHE_MB=1792` | [0175](../ai/tasks/0175-moe-pread-cache.md), [0177](../ai/tasks/0177-moe-speed.md) |
| 30B MoE, TTFT | **13,4 с** | `tools/chat.py --stats` | [0180](../ai/tasks/0180-cache-plus-batch.md) |
| 30B MoE, контейнер | **17,28 ГиБ** (18 554 374 000 Б) | `ls -l`, импорт | [0174](../ai/tasks/0174-stream-import.md) |
| Silero, фраза после prewarm | **0,04–0,18 с** | `/api/tts`, воркер | [0195](../ai/tasks/0195-silero-worker.md) |
| Silero, RSS worker'а | **0,84–0,93 ГБ** | `ru_maxrss` | [0195](../ai/tasks/0195-silero-worker.md) |
| EVA.app, размер бандла | **≈0,7 МБ** | `sh tools/build_eva_app.sh` | [0184](../ai/tasks/0184-native-voice-app.md), [0194](../ai/tasks/0194-stop-command.md) |
| 3B Q4, размеры | 1,92 ГиБ (Q4_0) / 2,245 ГиБ (Q4_1) | `ls -l runs/q4/` | [0155](../ai/tasks/0155-stream-q4-writer.md), [0160](../ai/tasks/0160-q4-asym-import.md) |
| 7B Q4, декод тёплый | **4,5 ток/с** (6 потоков) | `vtm run --stats` | [0161](../ai/tasks/0161-q4-7b-slow.md) |

## ВТМ: что есть

ВТМ — виртуальная троичная машина: C99-интерпретатор, Python-эталон и
компилятор графа в контейнер `.vtm` v0.2/v0.3. Спецификация — [vtm.md](vtm.md),
производительность — [vtm-perf.md](vtm-perf.md), устройство —
[src/vtm/README.md](../src/vtm/README.md).

Есть: контейнер v0.2 (u32) и v0.3 (u64-смещения, [ADR 0019](adr/0019-u64-offsets.md))
с тритами/масштабами/метой и **19 опкодами** (`TMV`, `MM`, `RMSN`, `ROPE`, `SM`,
`SILU`, `MUL`, `ADD`, `EMB`, `KVW`, `ATTN`, `SAMP`, `CPY`, `SCAL`, `HALT`, `MOE`,
`RMSH`, `CONV4`, `GDELTA` — сверяет `python3 tools/check_vtm_docs.py`);
BPE-токенизатор на C++17; текстовый путь `--text` и чат-шаблон Qwen;
сэмплинг; mmap-загрузка; SHA-256; C-ABI `libvtm.a`; сессии KV (VTMS v1);
трассировка; кэш префикса; фаззинг и санитайзеры; потоки в ядрах
(`VTM_THREADS=1..8`); батчевый prefill
([0112](../ai/tasks/0112-prefill-batch.md), [0114](../ai/tasks/0114-prefill-cli.md)).

**Бэкенды:** `auto` → Metal, если он собран и есть GPU, иначе SIMD (NEON/AVX2),
иначе scalar. Явно: `VTM_BACKEND=metal|hybrid|simd|scalar`. Metal считает
GEMV/GEMM kind 6/7 (Q4_K/Q6_K) и в фазе 3 — весь токен декода одним графом;
kind 5 (Q4_0/Q4_1), триты, MoE и плотные веса идут CPU-путём
([0192](../ai/tasks/0192-metal-gemv.md), [0196](../ai/tasks/0196-metal-phase2.md),
[0198](../ai/tasks/0198-hybrid-gpu-prefill.md),
[0204](../ai/tasks/0204-metal-phase3.md)). Ошибка Metal откатывает токен на CPU
и выключает бэкенд, а не портит ответ.

**Что именно не сделано** — честный список в [vtm.md](vtm.md)
§ «Что не реализовано».

## Q4, GGUF и MoE

Kind 5 — Q4_0/Q4_1 (4 бита + масштаб, min у Q4_1, группа 32, раскладка GGML),
kind 6/7 — нативные Q4_K/Q6_K ([ADR 0017](adr/0017-q4-kind.md),
[ADR 0018](adr/0018-q4k-native.md)); версия контейнера для kind 5 осталась 2.

- **Импорт из ollama:** `python3 tools/gguf_import.py import <blob> --out <dir>
  --format q4` (Q4_K_M → Q4_0/Q4_1), затем `python3 tools/vtm_compile.py
  --manifest <dir>/manifest.json --out <file>.vtm`. Флаги: `--q4-kind
  auto|q4_0|q4_1`, `--q4-scale-dtype fp32|fp16`, `--embed int8|trit|fp16`
  ([0153](../ai/tasks/0153-gguf-import.md), [0156](../ai/tasks/0156-q4-7b.md),
  [0160](../ai/tasks/0160-q4-asym-import.md)).
- **Размеры:** 3B Q4_0 — 2 063 237 920 Б (**1,92 ГиБ**), 3B Q4_1 —
  2 410 054 560 Б (**2,245 ГиБ**); 7B Q4_0 с fp16-масштабами и тритовым
  эмбеддингом — 4 113 979 440 Б (**3,831 ГиБ, 95,79%** от 4 ГиБ).
- **Качество:** Q4_1 снизил rel-RMS с 9,96% до **5,1%** по Linear 3B; текст на
  «Столица Франции?» стал «Столицей Франции является Париж» вместо мусора
  ([0160](../ai/tasks/0160-q4-asym-import.md)). Это постичь-квантование готовой
  Qwen, а не обученная модель.
- **7B и Q4_1 несовместимы:** +0,44 ГБ = 106% предела 4 ГиБ, поэтому 7B
  собирается `--q4-kind q4_0` ([0156](../ai/tasks/0156-q4-7b.md),
  [0160](../ai/tasks/0160-q4-asym-import.md)).
- **30B MoE** — отдельная ветка: контейнер v0.3 (u64), потоковый импорт
  ([0173](../ai/tasks/0173-u64-container.md),
  [0174](../ai/tasks/0174-stream-import.md)), операция `MOE`, эксперты через
  `pread` в LRU-кэш и I/O-пул ([0175](../ai/tasks/0175-moe-pread-cache.md),
  [0177](../ai/tasks/0177-moe-speed.md)), батчевый prefill и упаковка слота
  активаций ([0179](../ai/tasks/0179-moe-batched-prefill.md),
  [0181](../ai/tasks/0181-moe-batch-bigger.md)). Скорость и потолок — в
  [moe.md](moe.md); гибрид gated delta (Qwen3.5) — Python-эталон
  [qwen35.md](qwen35.md), C-часть ждёт.

## Экспорт и третизация

Холодная упаковка обученной Qwen в триты — **закрытый путь**, не «недоделанный»:

| Схема | 0,5B / 3B, ×bf16 | Читаемо | Задача |
|---|---|---|---|
| 1 трит (Linear) | ×3,9e5 / ×4,5e6 | нет | [0142](../ai/tasks/0142-cold-3b.md) |
| 1 трит + тритовая таблица | ×2,8e6 / ×1,7e7 | нет | [0136](../ai/tasks/0136-cold-parity.md) |
| 2 трита (2 бита) | ×217, декод вырождается | нет | [0148](../ai/tasks/0148-exp-2bit.md) |
| **int8** | **×1,004** | да | [0136](../ai/tasks/0136-cold-parity.md) |
| **Q4** | размер 1,92/2,245 ГиБ, паритет ядер 1e-6 | да | [0155](../ai/tasks/0155-stream-q4-writer.md) |
| **Триты после QAT** | — | цель | [0008](../ai/tasks/0008-qat-pilot-run.md) |

Выводы и оговорки — [quantization-verdict.md](quantization-verdict.md): урон
растёт с размером модели, главный источник — Linear, а не таблица словаря;
два трита стоят 3,2–3,45 бита и проигрывают int8. Размеры для 3B: Q4 ~1,9 ГБ,
int8 ~3,1 ГБ, QAT-триты ~0,69 ГБ.

Сквозная цепочка QAT → export → `.vtm` → C пройдена на крошечной модели:
val loss 5,5759 → 1,5704, `max|Δ|` C↔HF **2,1e-06**
([0095](../ai/tasks/0095-tiny-qat-cpu.md), [0099](../ai/tasks/0099-trit-embeddings.md)).

## Данные

**75 000 пар** в смеси `runs/mixture` (train 67 500 / val 7 500, seed 42):
код 45,4%, бизнес 35,0%, общее 17,4%, поведение 2,2%
(файл `runs/mixture/stats.json`; сборка —
`python3 tools/build_mixture.py --total 75000 --out runs/mixture`,
план — [data-plan.md](data-plan.md)).

Отдельно лежит исходный банк кодовых пар: **4008** строк в
[distill/out/code_pairs.jsonl](../distill/out/code_pairs.jsonl), **0,032 ₽/пара**
(≈128 ₽ за банк) — [0045](../ai/tasks/0045-code-data.md). Это не «вся смесь»:
4008 — один из источников слоя «код».

## QAT

Пилота на аренде не было: на машине нет torch/CUDA
([0008](../ai/tasks/0008-qat-pilot-run.md), статус `blocked`). Оценки памяти
([0047](../ai/tasks/0047-qat-3b-24gb.md)): 1.5B fp32-мастер + Adam8bit — 21,32 ГиБ;
3B bf16-мастер + Adam8bit — **19,93 ГиБ** (+4,07 до 24 ГиБ); 3B fp32-мастер —
31,43 ГиБ. Конфиги и политики (fp32-масштабы, тритовый эмбеддинг, `long_pairs`,
`max_steps`) — [qat.md](qat.md), проверены замерами
[0092](../ai/tasks/0092-qat-data-prep.md)–[0099](../ai/tasks/0099-trit-embeddings.md).

Не измерено: реальный пик VRAM, сходимость на 1.5B/3B, качество после QAT.
Данные для пилота теперь есть — 75 000 пар вместо 4008; рунбук аренды
([rental-runbook.md](rental-runbook.md)) считает шаги от 4008 и требует
пересчёта под смесь.

## Чаты, голос и клиенты

- **`tools/chat.py`** — REPL для трёх бэкендов: `--backend vtm|ollama|openai`,
  история диалога, обрезка по контексту, стриминг по токенам, `/reset`,
  `/system`, `/stats`, `/quit`. `/stats` показывает TTFT, prompt/completion,
  ток/с, стоимость, заполнение контекста; с `--think` — ещё и токены
  размышления ([0137](../ai/tasks/0137-chat-cli.md),
  [0200](../ai/tasks/0200-qwen-template-thinking.md)).
- **`tools/chat_server.py`** — только 127.0.0.1, один вопрос → два ответа
  рядом (ВТМ и ollama), SSE, без внешних CDN. Панель под чатом: размер модели
  на диске, параметры, KV, RSS-дельта, TTFT/prefill/декод, ток/с, load/RAM/диск,
  CPU-время ([0138](../ai/tasks/0138-chat-web.md),
  [0145](../ai/tasks/0145-chat-stats-panel.md)). Память диалога — на сервере
  ([0182](../ai/tasks/0182-chat-memory.md)), `/api/chat`, `/api/tts`,
  `/api/cancel`, `/api/reset`.
- **TTS-цепочка:** автовыбор — **ElevenLabs** (если задан ключ) → **Silero**
  (локально) → **`say`** (аварийно) → null; основной локальный голос — Silero.
  Silero — `baya` по умолчанию,
  `kseniya`/`xenia` в селекторах, `aidar` через API; worker
  поднимается один раз, фраза после prewarm — **0,04–0,18 с**, RSS
  **0,84–0,93 ГБ** ([0193](../ai/tasks/0193-silero-tts.md),
  [0195](../ai/tasks/0195-silero-worker.md),
  [0197](../ai/tasks/0197-switch-to-silero-only.md)). ElevenLabs: mp3,
  LRU-кэш, при 401/429/сети — автоматический откат на локальный голос
  ([0191](../ai/tasks/0191-elevenlabs-tts.md)).
- **Нормализация речи** — общая ступень до выбора движка: числа, %, ₽, даты,
  диапазоны ([0205](../ai/tasks/0205-tts-numbers.md)); знаки, сокращения,
  телефоны, email, хештеги ([0206](../ai/tasks/0206-tts-symbols.md));
  женский род только в первом лице ([0207](../ai/tasks/0207-feminine-speech.md)).
  Детали и таблицы — [hybrid.md](hybrid.md).
- **EVA.app** (macOS, ≈0,7 МБ) — статус-бар, ⌥Space, Apple Speech on-device,
  wake-word «Привет, Ева», follow-up после ответа, круг состояний, компактный
  круг 140 pt, «Стоп, Ева»/«хватит»/«тише» с эхоподавлением, голоса Silero
  ([0184](../ai/tasks/0184-native-voice-app.md),
  [0187](../ai/tasks/0187-wake-word.md),
  [0190](../ai/tasks/0190-followup-listening.md),
  [0194](../ai/tasks/0194-stop-command.md); полное описание —
  [voice-app.md](voice-app.md)). Чат остаётся стендом, продукт — приложение
  ([ADR 0020](adr/0020-clients-and-cloud.md)).
- **Компактный промпт по умолчанию**: `prompts/system.compact.md` (3 строки)
  вместо длинного `prompts/system.assistant.md`; `--system` переопределяет.
  Причина — квантованная 3B теряет следование инструкции на длинном промпте
  (коммит `a6f6863`, [0163](../ai/tasks/0163-int8-dot-prefill.md)).

## Плагины (в работе)

Решение — [ADR 0022](adr/0022-plugins.md): плагин это каталог `plugins/<id>/`
с манифестом `plugin.json` (`id`, `name`, `version`, `api: 1`, `entry`,
`capabilities`, `permissions`, `settings`), способности — `skill.fetch`,
`filter.run`, `tts.engine`, `command`. Ядро должно знать только «вызови скиллы,
подложи факты, прогони фильтры», а права ограничивать сеть/файлы.

Состояние: контракт и каркас `eva/plugins/` есть, каталоги первых плагинов
(`time`, `currency`, `stocks`, `ru-numbers`, `ru-gender`) пока пусты; включения
через `config/plugins.json` и API нет. Задача
[0209](../ai/tasks/0209-plugin-runtime.md) — `in_progress`, поэтому
[docs/plugins.md](plugins.md) описывает контракт, а не готовый API. До
закрытия 0209 рабочий путь расширения — скиллы `eva/skills`
([0208](../ai/tasks/0208-eva-tools.md), тоже `in_progress`).

## Эталон и golden

- `eval/golden_competence.jsonl` — **72** строки, 9 категорий × 8; проверки
  структурные ([golden.md](golden.md), [0124](../ai/tasks/0124-golden-competence.md)).
- **qwen2.5:3b — 65/72 = 90,3%** на сохранённых ответах после исправления
  чекера: было 60/72 = 83,3%, пять ложных провалов убраны
  ([0140](../ai/tasks/0140-checker-morphology.md)).
  `python3 tools/eval_competence.py --answers runs/eval/ollama-qwen2.5-3b-live.jsonl`.
- **Наша 3B Q4_1 с компактным промптом** — 20 вопросов: reasoning 8/8,
  marketing 8/8, clarify 2/8 (4 строки clarify не прогнаны — таймауты на
  prefill). Файл `runs/eval/vtm-3b-q4_1-compact.jsonl`, оценка:
  `python3 tools/eval_competence.py --answers runs/eval/vtm-3b-q4_1-compact.jsonl`.
  Это замер на 20 строках, а не полный golden.
- **Гибрид** (`python3 tools/eval_hybrid.py`, модели-заглушки): на каталоге
  45 строк — маршруты 45/45, скиллы 44/45, **10,42 ₽**/1000. На историческом
  наборе 80 строк (`--golden eval/hybrid_golden.jsonl`) — 80/80, 44/80,
  **128,94 ₽** ([0105](../ai/tasks/0105-golden-classes.md),
  [0119](../ai/tasks/0119-prompt-shorten.md)).

## Тесты

```
$ python3 -m unittest discover -s tests -q
Ran 2570 tests
OK (skipped=71)                                  # 0200, 2026-11-26

$ cd src/vtm && make test                        # test_vtm, test_vtm_api,
                                                 # test_vtm_session, test_sha256,
                                                 # bench_backend --check
$ python3 tools/check_vtm_docs.py                # 19 опкодов: OK
$ python3 tools/check_tasks.py                   # 201 задача, id уникальны
$ python3 tools/check_docs_dates.py              # 31 документ, ссылки целы
```

71 скип — не сетевые: `torch не установлен` (GPU-путь), тяжёлые прогоны под
переменными (`VTM_BENCH_SHAPE_HEAVY`, `VTM_GGUF_7B`, `EVA_*`), TTS-бинарники.
CI — [.github/workflows/ci.yml](../.github/workflows/ci.yml): ubuntu + macos,
C-сборка, Python-набор, реестр задач, документация VTM и свежесть docs.
Локальный аналог — `python3 tools/ci.py`. Число тестов — из отчёта
[0200](../ai/tasks/0200-qwen-template-thinking.md); после него добавились
0205/0206/0207 (и в работе 0209), поэтому число перепроверяется `tools/ci.py`.

## Заблокировано и открыто

| Что | Состояние | Почему / что от этого зависит |
|---|---|---|
| Пилот QAT на аренде [0008](../ai/tasks/0008-qat-pilot-run.md) | `blocked` | нет GPU: 4090 24 ГБ на 6–8 ч, 600–900 ₽; решение за Юрием. Нужен для обученной тернарной модели, качества и реального пика VRAM |
| Банк вопросов из каталога [0007](../ai/tasks/0007-catalog-questions.md) | `blocked` | каталог демонстрационный (13 позиций) |
| 32B в ВТМ [0141](../ai/tasks/0141-vtm-u64-offsets.md) | `pending` | u64-смещения уже есть в v0.3 (ADR 0019), осталось решить, нужен ли 32B в ВТМ |
| [0127](../ai/tasks/0127-cabi-prefill-ms.md)–[0133](../ai/tasks/0133-pairstore-index.md) | `pending` | C-ABI `prefill_ms`, триты через `trit_store`, ленивый int8-эмбеддинг, sidecar логитов, `max_top_k`, индекс PairStore |
| Смешанная точность [0146](../ai/tasks/0146-exp-mixed.md), AWQ/GPTQ [0147](../ai/tasks/0147-exp-smart.md), разреженный второй трит [0149](../ai/tasks/0149-sparse-trit.md) | `in_progress` | эксперименты остановлены из-за нехватки RAM; числа не получены |
| Скиллы свежих данных [0208](../ai/tasks/0208-eva-tools.md) | `in_progress` | время/курсы/котировки ещё переносятся; до этого — старые скиллы |
| Рантайм плагинов [0209](../ai/tasks/0209-plugin-runtime.md) | `in_progress` | манифест/реестр есть, плагинов и включения нет; см. [plugins.md](plugins.md) |
| Декод: память и параллелизм [0201](../ai/tasks/0201-decode-memory-parallelism.md) | `in_progress` | потолок потоков по P-ядрам, векторный SiLU; дальше — меньше памяти на токен |
| 7B QAT | не начат | отдельной задачи в реестре нет; сначала 0008, затем 3B/7B |

## Дальше

- **0008, QAT на аренде** — после него появляются числа «до/после» для
  1.5B/3B и проверка тезиса «триты не хуже int8» на качестве, а не на паритете.
- **30B MoE быстрее** — Q3-контейнер (−25% байт) и int8-батчевое ядро kind 6/7;
  сейчас упор в 4,09 ГиБ `pread` на прогон ([0177](../ai/tasks/0177-moe-speed.md),
  [moe.md](moe.md)).
- **Плагины (0209)** — довести манифест/включение/права и первые плагины;
  после этого ядро перестаёт знать про конкретные данные.
- **Клиенты** — iPhone 11 и Android по тому же HTTP/SSE-контракту
  ([ADR 0020](adr/0020-clients-and-cloud.md)); телефон всегда тонкий клиент.
- **0141, u64/шардинг** — только если понадобится 32B в ВТМ; v0.3 уже умеет
  u64, вопрос в цене и надобности.
- **0146/0147/0149** — эксперименты по точности без GPU: им нужна RAM, а не
  аренда; после освобождения машины.

## Риски

- **Числа скорости сняты на загруженной машине** (load average 6–41 в отчётах
  0157–0204). Отношения устойчивы (Metal 1,4–2,0× к CPU), абсолютные ток/с
  гуляют до 2×; перед релизом перемерить на свободной.
- **Metal ускоряет не всё.** Граф поддерживает kind 4/6/7 при `n == 1`; Q4_1
  (kind 5), триты, MoE и плотные веса идут CPU-путём. Числа 24–30 относятся к
  `3b-q4k.vtm`, а не к `3b-q4_1.vtm` ([0204](../ai/tasks/0204-metal-phase3.md)).
- **30B MoE — режим «работает», а не продукт.** 1,88 ток/с и TTFT 13,4 с
  упираются в диск; на 16 ГиБ модель не держится в page cache
  ([0177](../ai/tasks/0177-moe-speed.md), [0180](../ai/tasks/0180-cache-plus-batch.md)).
- **3B Q4_1 — не обученная модель.** rel-RMS 5,1% и «Париж» на одном промпте
  не заменяют golden: clarify 2/8, часть строк уходит в таймаут на prefill.
  Качество лечится QAT (0008), а не форматом.
- **int8-dot не годится для качества по умолчанию:** ошибка логитов 1,0e-1
  при допуске 1e-4; по умолчанию выключен (`VTM_Q4_I8=0`), ядро даёт 2,2×
  ([0163](../ai/tasks/0163-int8-dot-prefill.md)).
- **7B Q4 зависит от RAM:** если 3,83 ГиБ вытесняются, каждый токен снова
  читает файл — пол ~1,5 с/токен ([0161](../ai/tasks/0161-q4-7b-slow.md)).
- **Q4_1 и 7B несовместимы** (106% предела), а Q4_0 у 7B — компромисс по
  качеству (тритовый эмбеддинг, rel-RMS 43,5% против int8 0,65%,
  [0156](../ai/tasks/0156-q4-7b.md)).
- **Плагины — код, который мы запускаем.** Permissions — best-effort, полной
  песочницы нет; для облака нужны «чьи плагины, чьи права»
  ([ADR 0022](adr/0022-plugins.md)).
- **Документация стареет быстрее кода.** С 0164 дата в каждом документе и
  проверка 30 дней в CI; ADR — исключение (дата решения историческая).
  Если документ противоречит коду — верьте коду и `tools/check_vtm_docs.py`.
- **Метаданные задач расходятся:** у части задач отчёт пуст, а статус `done`
  (например, [0207](../ai/tasks/0207-feminine-speech.md)); числа взяты из
  отчётов и кода. Реестр здесь — источник статуса.

## Как перепроверить

```bash
python3 tools/eval_hybrid.py                                  # 45/45, 44/45, 10,42 ₽
python3 tools/eval_hybrid.py --golden eval/hybrid_golden.jsonl # 80/80, 44/80, 128,94 ₽
python3 -m unittest discover -s tests -q                      # 2570 tests, OK (skipped=71)
cd src/vtm && make test                                       # C-набор и паритет Q4
cd ../.. && python3 tools/check_vtm_docs.py                   # 19 опкодов: OK
python3 tools/check_tasks.py                                  # 201 задача
python3 tools/check_docs_dates.py                             # 31 документ, ссылки
python3 tools/eval_competence.py --answers runs/eval/ollama-qwen2.5-3b-live.jsonl
python3 tools/eval_competence.py --answers runs/eval/vtm-3b-q4_1-compact.jsonl
wc -l runs/mixture/train.jsonl runs/mixture/val.jsonl         # 67500 / 7500
ls -l runs/q4/3b-q4_1.vtm runs/q4/3b.vtm                      # 2,245 / 1,92 ГиБ

# 3B Q4_K: Metal против CPU (парный замер)
cd src/vtm && ./bench_backend --metal ../../runs/q4/3b-q4k.vtm --metal-new 16

# 30B MoE: декод 1,88 ток/с с кэшем экспертов 1792 МБ
VTM_Q4K_CHECK_OFF=1 VTM_MOE_CACHE_MB=1792 ./vtm run \
    ../../runs/q4/qwen3-30b-a3b.vtm --tokens 151644,872,198 --max-new 4 --stats

# Silero: фраза после prewarm
curl -s -o /tmp/eva.wav -X POST localhost:8765/api/tts \
    -d '{"text":"Проверка голоса","voice":"baya"}'
```

## Куда дальше

- [README.md](README.md) — индекс документации с датами.
- [architecture.md](architecture.md) — устройство системы: клиенты, сервер, ВТМ, плагины.
- [voice-app.md](voice-app.md) — приложение EVA.app и голосовые режимы.
- [moe.md](moe.md) — 30B MoE: числа, потолок, что дальше.
- [plugins.md](plugins.md) — контракт плагинов (в работе).
- [roadmap.md](roadmap.md) — фазы и критерии продукта.
- [adr/README.md](adr/README.md) — архитектурные решения.
- [../ai/tasks/index.json](../ai/tasks/index.json) — реестр задач и статусы.
