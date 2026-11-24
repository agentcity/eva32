# Состояние проекта EVA32

> Актуально на 2026-11-24.

Снимок на **2026-11-24**. У каждого числа — задача или команда, которой оно
получено; «не измерено» значит ровно это. Реестр задач —
[`ai/tasks/index.json`](../ai/tasks/index.json): **156 задач**, из них 141 `done`,
8 `pending`, 5 `in_progress`, 2 `blocked` (`python3 tools/check_tasks.py`).

## Коротко

- **Q4 в рантайме, а не гипотеза.** 3B Q4 загружается и считает: **2,245 ГиБ**
  (Q4_1) и **1,92 ГиБ** (Q4_0), 7B Q4 — **3,831 ГиБ = 95,79%** предела 4 ГиБ
  ([0155](../ai/tasks/0155-stream-q4-writer.md),
  [0156](../ai/tasks/0156-q4-7b.md), [0160](../ai/tasks/0160-q4-asym-import.md);
  формат — [ADR 0017](adr/0017-q4-kind.md), предел — [ADR 0016](adr/0016-vtm-4gib-limit.md)).
- **Скорость выросла в разы, но prefill остаётся узким местом.** NEON Q4 —
  **7,8×** на ядре ([0157](../ai/tasks/0157-q4-neon.md)), GEMV lm_head — **10,9×**
  ([0158](../ai/tasks/0158-dense-mm-simd.md)), батчевый Q4-prefill — **1,1–1,5×**
  (это потолок арифметики, [0162](../ai/tasks/0162-q4-prefill-batch.md)).
  Декод 3B Q4 — **7–8 ток/с** в сессии; 7B Q4 — **4,5 ток/с** тёплым на 6 потоках
  ([0161](../ai/tasks/0161-q4-7b-slow.md)).
- **Холодная третизация непригодна** — ×3,9e5 (0,5B) и ×4,5e6 (3B) к bf16,
  два трита ×217 и проигрывают int8 (×1,004). Триты имеют смысл только после
  QAT ([quantization-verdict.md](quantization-verdict.md),
  [0142](../ai/tasks/0142-cold-3b.md), [0143](../ai/tasks/0143-embed-damage.md),
  [0148](../ai/tasks/0148-exp-2bit.md)).
- **Данные — 75 000 пар**: код 45,4%, бизнес 35,0%, общее 17,4%, поведение 2,2%
  (файл `runs/mixture/stats.json`,
  `python3 tools/build_mixture.py --total 75000 --out runs/mixture`).
- **Чаты работают на трёх бэкендах** — CLI `tools/chat.py` (стриминг, история,
  `/stats`) и локальная страница `tools/chat_server.py` (два ответа рядом,
  панель диск/RAM/CPU). Подробности — раздел «Чаты».
- **Эталон:** qwen2.5:3b — **65/72 = 90,3%** после исправления чекера
  ([0140](../ai/tasks/0140-checker-morphology.md)). Наша 3B Q4_1 с компактным
  промптом: reasoning **8/8**, marketing **8/8**, clarify **2/8** (20 вопросов,
  `runs/eval/vtm-3b-q4_1-compact.jsonl`).
- **Тесты:** **2305** Python-тестов, `OK (skipped=61)`, CI зелёный на ubuntu и
  macos (`python3 -m unittest discover -s tests`).

## ВТМ: что есть

ВТМ — виртуальная троичная машина: C99-интерпретатор, Python-эталон и
компилятор графа в контейнер `.vtm` v0.2. Спецификация — [vtm.md](vtm.md),
производительность — [vtm-perf.md](vtm-perf.md), устройство —
[src/vtm/README.md](../src/vtm/README.md).

Есть: контейнер v0.2 с тритами/масштабами/метой и **15 опкодов** (`TMV`, `MM`,
`RMSN`, `ROPE`, `SM`, `SILU`, `MUL`, `ADD`, `EMB`, `KVW`, `ATTN`, `SAMP`, `CPY`,
`SCAL`, `HALT` — сверяет `python3 tools/check_vtm_docs.py`); BPE-токенизатор на
C++17; текстовый путь `--text` и чат-шаблон Qwen `--chat`; сэмплинг;
mmap-загрузка; SHA-256; C-ABI `libvtm.a`; сессии KV (VTMS v1); трассировка;
кэш префикса; фаззинг и санитайзеры; потоки в ядрах (`VTM_THREADS=1..8`);
батчевый prefill ([0112](../ai/tasks/0112-prefill-batch.md),
[0114](../ai/tasks/0114-prefill-cli.md)). Что именно не сделано — честный
список в [vtm.md](vtm.md) § «Что не реализовано».

| Что измерено | Число | Чем проверено | Задача |
|---|---:|---|---|
| 3B Q4, декод в сессии | **7–8 ток/с** | `tools/chat.py --backend vtm` | [0162](../ai/tasks/0162-q4-prefill-batch.md) |
| 7B Q4, декод тёплый | **4,5 ток/с** (6 потоков), 1,1 (1 поток) | `vtm run … --stats` | [0161](../ai/tasks/0161-q4-7b-slow.md) |
| 3B Q4, TTFT 512 токенов | 206 с в холодном CLI, 59 с в процессе | `bench_backend --prefill-ttft` | [0162](../ai/tasks/0162-q4-prefill-batch.md) |
| Q4-ядро NEON | **7,8×** (151936×896, 91,6 GFLOP/s) | `bench_backend --q4 --check` | [0157](../ai/tasks/0157-q4-neon.md) |
| GEMV плотного lm_head | **10,9×** (637,4 → 58,4 мс) | `bench_backend` | [0158](../ai/tasks/0158-dense-mm-simd.md) |
| Батчевый Q4-prefill | 1,1–1,5×, ядро 63,6 → 93,3 GFLOP/s | `bench_backend --q4-batch` | [0162](../ai/tasks/0162-q4-prefill-batch.md) |
| int8-dot (SDOT) | 2,21× к f32-батчу; **по умолчанию выключен** | `bench_backend --q4-batch` | [0163](../ai/tasks/0163-int8-dot-prefill.md) |
| Паритет C↔Python | 1,435e-6 (3B), 2,147e-6 (7B, срез) | `--prefill-check` | [0155](../ai/tasks/0155-stream-q4-writer.md), [0156](../ai/tasks/0156-q4-7b.md) |

**Не измерено:** ВТМ на обученных тритовых весах (QAT ждёт GPU, 0008);
AVX2/VNNI на живом x86 (только кросс-компиляция — [0159](../ai/tasks/0159-gemv-to-backend.md),
[0163](../ai/tasks/0163-int8-dot-prefill.md)); 32B в ВТМ (не влезает, 0141).
Windows и big-endian закрыты честным отказом.

## Q4 и GGUF

Kind 5 — четвёртый числовой вид контейнера: 4 бита на вес плюс масштаб (и
`min` у Q4_1) на группу 32, раскладка GGML Q4_0; версия контейнера осталась 2.
Формат — [ADR 0017](adr/0017-q4-kind.md), разбор сегмента — [vtm.md](vtm.md)
§ «Q4 (kind 5, задача 0151)».

- **Импорт из ollama:** `python3 tools/gguf_import.py import <blob> --out <dir>
  --format q4` (Q4_K_M → Q4_0/Q4_1), затем `python3 tools/vtm_compile.py
  --manifest <dir>/manifest.json --out <file>.vtm`. Флаги:
  `--q4-kind auto|q4_0|q4_1`, `--q4-scale-dtype fp32|fp16`,
  `--embed int8|trit|fp16` ([0153](../ai/tasks/0153-gguf-import.md),
  [0156](../ai/tasks/0156-q4-7b.md), [0160](../ai/tasks/0160-q4-asym-import.md)).
- **Размеры:** 3B Q4_0 — 2 063 237 920 Б (**1,92 ГиБ**), 3B Q4_1 —
  2 410 054 560 Б (**2,245 ГиБ**); 7B Q4_0 с fp16-масштабами и тритовым
  эмбеддингом — 4 113 979 440 Б (**3,831 ГиБ, 95,79%** от 4 ГиБ).
- **Качество:** Q4_1 снизил rel-RMS с 9,96% до **5,1%** по Linear 3B; текст на
  «Столица Франции?» стал «Столицей Франции является Париж» вместо мусора
  ([0160](../ai/tasks/0160-q4-asym-import.md)). Это постичь-квантование готовой
  Qwen, а не обученная модель: качество 3B Q4_1 в golden — раздел «Эталон».
- **7B и Q4_1 несовместимы:** +0,44 ГБ = 106% предела 4 ГиБ, поэтому 7B
  собирается `--q4-kind q4_0` ([0156](../ai/tasks/0156-q4-7b.md),
  [0160](../ai/tasks/0160-q4-asym-import.md)).
- **Скорость:** cold-page провал 7B был не в ядрах, а в фолтах mmap
  (0,26 ГБ/с): добавлена упреждающая подкачка (0161), 7B холодный токен
  17,5 → 6,4–7,8 с. Физический пол при вытеснении страниц — ~1,5 с/токен.

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

## Чаты

- **`tools/chat.py`** — REPL для трёх бэкендов: `--backend vtm|ollama|openai`,
  история диалога, обрезка по контексту, стриминг по токенам, `/reset`,
  `/system`, `/stats`, `/quit`. `/stats` показывает TTFT, prompt/completion,
  ток/с, стоимость, заполнение контекста. 83 теста на моках
  ([0137](../ai/tasks/0137-chat-cli.md)).
- **`tools/chat_server.py`** — только 127.0.0.1, один вопрос → два ответа
  рядом (ВТМ и ollama), SSE, без внешних CDN. Панель под чатом: размер модели
  на диске, параметры, KV, RSS-дельта, TTFT/prefill/декод, ток/с, load/RAM/диск,
  CPU-время ([0138](../ai/tasks/0138-chat-web.md),
  [0145](../ai/tasks/0145-chat-stats-panel.md)).
- **Компактный промпт по умолчанию**: `prompts/system.compact.md` (3 строки)
  вместо длинного `prompts/system.assistant.md`; `--system` переопределяет.
  Причина — квантованная 3B теряет следование инструкции на длинном промпте
  (коммит `a6f6863`, [0163](../ai/tasks/0163-int8-dot-prefill.md)).

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
  Это замер на 20 строках, а не полный golden: числа по остальным категориям
  ещё не сняты.
- **Гибрид** (`python3 tools/eval_hybrid.py`, модели-заглушки): на каталоге
  45 строк — маршруты 45/45, скиллы 44/45, **10,42 ₽**/1000. На историческом
  наборе 80 строк (`--golden eval/hybrid_golden.jsonl`) — 80/80, 44/80,
  **128,94 ₽** (повторяет [0105](../ai/tasks/0105-golden-classes.md),
  [0119](../ai/tasks/0119-prompt-shorten.md)).

## Тесты

```
$ python3 -m unittest discover -s tests
Ran 2305 tests in 57.6s
OK (skipped=61)                                  # exit 0

$ cd src/vtm && make test                        # test_vtm, test_vtm_api,
                                                 # test_vtm_session, test_sha256,
                                                 # bench_backend --check
$ python3 tools/check_vtm_docs.py                # 15 опкодов, 25 ссылок: OK
$ python3 tools/check_tasks.py                   # 156 задач, id уникальны
$ python3 tools/check_docs_dates.py              # даты и ссылки документации
```

61 скип — не сетевые: `torch не установлен` (GPU-путь), тяжёлые прогоны под
переменными (`VTM_BENCH_SHAPE_HEAVY`, `VTM_GGUF_7B`, `EVA_*`), TTS-бинарники.
CI — [.github/workflows/ci.yml](../.github/workflows/ci.yml): ubuntu + macos,
C-сборка, Python-набор, реестр задач, документация VTM и свежесть docs.
Локальный аналог — `python3 tools/ci.py`.

## Заблокировано и открыто

| Что | Состояние | Почему / что от этого зависит |
|---|---|---|
| Пилот QAT на аренде [0008](../ai/tasks/0008-qat-pilot-run.md) | `blocked` | нет GPU: 4090 24 ГБ на 6–8 ч, 600–900 ₽; решение за Юрием. Нужен для обученной тернарной модели, качества и реального пика VRAM |
| Банк вопросов из каталога [0007](../ai/tasks/0007-catalog-questions.md) | `blocked` | каталог демонстрационный (13 позиций) |
| 32B в ВТМ [0141](../ai/tasks/0141-vtm-u64-offsets.md) | `pending` | u32-смещения: 32B = 6,92 ГиБ > 4 ГиБ (ADR 0016); сначала скорость (int8-dot/AVX-512), потом формат |
| int8-dot [0163](../ai/tasks/0163-int8-dot-prefill.md) | `in_progress` | ядро 2,21×, но ошибка логитов 1,0e-1 (режим 2) / 4,0 (режим 1); по умолчанию выключен (`VTM_Q4_I8=0`) |
| Смешанная точность [0146](../ai/tasks/0146-exp-mixed.md), AWQ/GPTQ [0147](../ai/tasks/0147-exp-smart.md), разреженный второй трит [0149](../ai/tasks/0149-sparse-trit.md) | `in_progress` | эксперименты остановлены из-за нехватки RAM; числа не получены |
| 7B QAT | не начат | отдельной задачи в реестре нет; сначала 0008, затем 3B/7B |
| [0127](../ai/tasks/0127-cabi-prefill-ms.md)–[0133](../ai/tasks/0133-pairstore-index.md) | `pending` | C-ABI `prefill_ms`, триты через `trit_store`, ленивый int8-эмбеддинг, sidecar логитов, `max_top_k`, индекс PairStore |

## Дальше

- **0163, int8-dot** — либо довести точность до допуска (два прохода с
  коррекцией), либо закрыть как «ядро есть, не дефолт»; это единственный
  оставшийся путь к prefill быстрее арифметики f32.
- **0008, QAT на аренде** — после него появляются числа «до/после» для
  1.5B/3B и проверка тезиса «триты не хуже int8» на качестве, а не на паритете.
- **0141, u64/шардинг** — только если понадобится 32B в ВТМ; сейчас
  осознанно отложено.
- **0146/0147/0149** — эксперименты по точности без GPU: им нужна RAM, а не
  аренда; после освобождения машины.
- **7B QAT** — после 0008; пока 7B живёт только как Q4_0 из GGUF.

## Риски

- **Числа скорости сняты на загруженной машине** (load average 14–32 в
  отчётах 0157–0163). Отношения устойчивы, абсолютные ток/с гуляют до 2×;
  перед релизом перемерить на свободной.
- **3B Q4_1 — не обученная модель.** rel-RMS 5,1% и «Париж» на одном промпте
  не заменяют golden: clarify 2/8, часть строк уходит в таймаут на prefill.
  Качество лечится QAT (0008), а не форматом.
- **int8-dot не годится для качества по умолчанию:** ошибка логитов 1,0e-1
  при допуске 1e-4, argmax совпадает только у двухпроходного режима.
- **7B Q4 зависит от RAM:** если 3,83 ГиБ вытесняются, каждый токен снова
  читает файл — пол ~1,5 с/токен (0161).
- **Q4_1 и 7B несовместимы** (106% предела), а Q4_0 у 7B — компромисс по
  качеству (тритовый эмбеддинг, rel-RMS 43,5% против int8 0,65%, 0156).
- **Документация стареет быстрее кода.** С 0164 дата в каждом документе и
  проверка 30 дней в CI; ADR — исключение (дата решения историческая).
  Если документ противоречит коду — верьте коду и `tools/check_vtm_docs.py`.
- **Метаданные задач расходятся:** у [0161](../ai/tasks/0161-q4-7b-slow.md) в
  файле `status: in_progress`, в реестре — `done`, отчёт пуст; числа взяты из
  коммита `8323f6f`. У [0145](../ai/tasks/0145-chat-stats-panel.md) отчёт
  заполнен, а в реестре `in_progress`. Реестр здесь — источник статуса.

## Как перепроверить

```bash
python3 tools/eval_hybrid.py                                  # 45/45, 44/45, 10,42 ₽
python3 tools/eval_hybrid.py --golden eval/hybrid_golden.jsonl # 80/80, 44/80, 128,94 ₽
python3 -m unittest discover -s tests                         # Ran 2305 tests, OK (skipped=61)
cd src/vtm && make test                                       # C-набор и паритет Q4
cd ../.. && python3 tools/check_vtm_docs.py                   # 15 опкодов, 24 ссылки
python3 tools/check_tasks.py                                  # 156 задач
python3 tools/check_docs_dates.py                             # даты и ссылки docs/
python3 tools/eval_competence.py --answers runs/eval/ollama-qwen2.5-3b-live.jsonl
python3 tools/eval_competence.py --answers runs/eval/vtm-3b-q4_1-compact.jsonl
wc -l runs/mixture/train.jsonl runs/mixture/val.jsonl         # 67500 / 7500
ls -l runs/q4/3b-q4_1.vtm runs/q4/3b.vtm                      # 2,245 / 1,92 ГиБ
cd src/vtm && ./vtm run ../../runs/q4/3b-q4_1.vtm --text "Столица Франции?" \
    --tokenizer ../../runs/q4/tokenizer.json --chat --max-new 32 --stats
```

## Куда дальше

- [README.md](README.md) — индекс документации с датами.
- [roadmap.md](roadmap.md) — фазы и критерии продукта.
- [work-plan.md](work-plan.md) — очерёдность работ и гейты.
- [adr/README.md](adr/README.md) — архитектурные решения.
- [../ai/tasks/index.json](../ai/tasks/index.json) — реестр задач и статусы.
