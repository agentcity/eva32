# EVA32

> Актуально на 2026-11-27.

Бандл модели для чата orioncode: веса (ссылка + ревизия), промпт, конфиг генерации, сервинг, eval.

Чего здесь нет и не должно быть: кода чата и Symfony (orioncode), RAG-индекса и документов (pgvector),
секретов, весов в git.

## Что сейчас (2026-11-26)

- **ВТМ считает 3B Q4 на CPU и на Metal.** 3B Q4_K — **24–30 ток/с** на Metal
  (режим `auto`), **16–20** на CPU simd; prefill на GPU **1,7–2,4×** быстрее CPU
  ([0204](ai/tasks/0204-metal-phase3.md), [0198](ai/tasks/0198-hybrid-gpu-prefill.md),
  [ADR 0021](docs/adr/0021-metal-gpu-backend.md)). Размеры: 3B Q4_1 —
  2,245 ГиБ, Q4_0 — 1,92 ГиБ ([0155](ai/tasks/0155-stream-q4-writer.md),
  [0160](ai/tasks/0160-q4-asym-import.md)); 7B Q4 — 3,831 ГиБ = 95,79% предела
  4 ГиБ ([0156](ai/tasks/0156-q4-7b.md), [ADR 0016](docs/adr/0016-vtm-4gib-limit.md)).
- **30B MoE работает на 16 ГиБ.** Qwen3-30B-A3B — **17,28 ГиБ**, **1,88 ток/с**,
  TTFT **13,4 с**: эксперты читаются `pread` в кэш 1792 МБ, I/O-пул параллельно
  со счётом ([0175](ai/tasks/0175-moe-pread-cache.md),
  [0177](ai/tasks/0177-moe-speed.md), [0180](ai/tasks/0180-cache-plus-batch.md),
  [docs/moe.md](docs/moe.md)).
- **Голос — Silero.** `baya` (по умолчанию), `kseniya`, `xenia`: **0,04–0,18 с** на
  фразу после prewarm, worker ≈0,9 ГБ ([0193](ai/tasks/0193-silero-tts.md),
  [0195](ai/tasks/0195-silero-worker.md)); ElevenLabs — по ключу,
  `say` — аварийно; прежние голосовые пресеты удалены
  ([0197](ai/tasks/0197-switch-to-silero-only.md)).
- **EVA.app — 0,7 МБ, всё локально:** ⌥Space, wake-word «Привет, Ева»,
  follow-up, круг состояний, компактный режим, «Стоп, Ева»
  ([0184](ai/tasks/0184-native-voice-app.md), [0186](ai/tasks/0186-eva-circle-visual.md),
  [0187](ai/tasks/0187-wake-word.md), [0188](ai/tasks/0188-compact-floating-circle.md),
  [0190](ai/tasks/0190-followup-listening.md), [0194](ai/tasks/0194-stop-command.md);
  [ADR 0020](docs/adr/0020-clients-and-cloud.md)).
- **Шаблон Qwen и размышления.** ChatML применяется по `tokenizer_config.json`,
  `<think>` по умолчанию выключен; `--reasoning-budget N` ограничивает
  рассуждение и **досказывает ответ**, а не теряет его
  ([0200](ai/tasks/0200-qwen-template-thinking.md)).
- **Речь нормализуется до синтеза:** числа, проценты, даты, телефоны, email,
  знаки и сокращения ([0205](ai/tasks/0205-tts-numbers.md),
  [0206](ai/tasks/0206-tts-symbols.md)); о себе Ева говорит в женском роде
  ([0207](ai/tasks/0207-feminine-speech.md)).
- **Плагины — в работе** (0209): каталог `plugins/<id>/` с `plugin.json`,
  способности `skill.fetch`/`filter.run`/`tts.engine`/`command`, права
  `permissions` ([ADR 0022](docs/adr/0022-plugins.md), [docs/plugins.md](docs/plugins.md)).
  Пока это контракт, а не готовый набор.
- **Данные — 75 000 пар** для QAT: код 45,4%, бизнес 35,0%, общее 17,4%, поведение 2,2%
  (`runs/mixture/stats.json`); сам QAT ждёт GPU ([0008](ai/tasks/0008-qat-pilot-run.md)).
- **Эталон:** qwen2.5:3b — 65/72 = 90,3% после исправления чекера
  ([0140](ai/tasks/0140-checker-morphology.md)); наша 3B Q4_1 с компактным промптом —
  20 вопросов: reasoning 8/8, marketing 8/8, clarify 2/8
  ([0139](ai/tasks/0139-eval-runner.md)).
- **Тесты:** Python-набор, C-набор и UBSan, CI зелёный на ubuntu и macos. Полная
  сводка — [docs/status.md](docs/status.md), индекс документации с датами —
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
- **[Полный список команд — docs/commands.md](docs/commands.md)** — что Ева умеет, на какие фразы и как отвечает; чего не умеет и что делать при сбое.
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

## Как работать с чатом

Три входа: нативное приложение **EVA.app** (голос и текст), веб-стенд (две модели
рядом) и терминал (REPL). Все три ходят в один сервер `tools/chat_server.py` и умеют
три бэкенда: локальная ВТМ, Ollama (OpenAI-совместимый) и внешний API.

### EVA.app: ⌥Space, «Привет, Ева» и «Стоп, Ева»

Приложение-компаньон к серверу: статус-бар, ответ печатается из SSE и по желанию
звучит. Всё локально, распознавание — Apple Speech on-device.

```bash
sh tools/build_eva_app.sh          # build/EVA.app (≈0,7 МБ, нужен swiftc)
sh tools/build_eva_app.sh --run    # собрать и открыть
```

| Действие | Что происходит |
|---|---|
| **⌥Space** | начать/закончить слушать; текст уходит в чат |
| **«Привет, Ева»** | wake-word в `idle` (галочка в настройках, по умолчанию выключена) |
| **follow-up** | после ответа микрофон открывается сам: следующий вопрос — без фразы; 12 с тишины → снова ожидание |
| **круг состояний** | `idle`/`listening`/`thinking`/`speaking`/`error` — видно, что делает Ева |
| **«Свернуть» / ⌘M** | остаётся компактный круг 140 pt поверх окон; клик — развернуть, двойной — слушать |
| **«Стоп, Ева» / «хватит» / «тише»** | глушит озвучку и отменяет генерацию (`POST /api/cancel`) |
| **«Стоп»** | то же кнопкой |

Подробности, настройки и самопроверки — [docs/voice-app.md](docs/voice-app.md);
решение о клиентах и облаке — [ADR 0020](docs/adr/0020-clients-and-cloud.md).
Голоса — `baya` (по умолчанию), `kseniya`, `xenia`; `aidar` доступен через API.

### Веб: два ответа рядом

```bash
ollama serve                                   # нужна для колонки Ollama
python3 tools/chat_server.py --port 8765       # http://127.0.0.1:8765/
```

Что на странице:

- **слева** — наша ВТМ, **справа** — Ollama; вопрос уходит в обе колонки сразу.
- **«Загрузить»/«Выгрузить»** в шапке колонки: загрузка прогревает веса, чтобы
  TTFT не платил за чтение файла (у Ollama кнопки неактивны — её памятью управляет
  её процесс).
- **панель под чатом** — размер файла, факт загрузки, RSS процесса, TTFT/prefill/
  декод, ток/с, CPU-время, load average, свободная RAM и диск.
- **системный промпт** — по умолчанию `prompts/system.compact.md` (3 строки) для
  квантованных моделей; боевой `prompts/system.assistant.md` (6 правил) — для сильных
  моделей: запустите с `--no-system-prompt` и подайте свой промпт из интерфейса или
  поменяйте файл. В CLI выбор — `--system путь`.
- `--keep-alive 300` — выгружать модель после 5 минут простоя (по умолчанию
  модель держится в памяти, пока сервер жив).

Сервер слушает только `127.0.0.1`, внешних запросов не делает.

### Терминал: REPL с историей и метриками

```bash
# локальная ВТМ (нужен tokenizer.json рядом с моделью)
python3 tools/chat.py --backend vtm \
    --model runs/q4/3b-q4k.vtm \
    --tokenizer runs/q4/tokenizer.json

# Ollama
python3 tools/chat.py --backend ollama --model qwen2.5:3b

# один вопрос без REPL (удобно для скриптов)
python3 tools/chat.py --backend vtm --model runs/q4/3b-q4k.vtm \
    --tokenizer runs/q4/tokenizer.json --once "Столица Франции?" --stats
```

Команды в REPL: `/reset`, `/system [текст|@файл]`, `/stats`, `/help`, `/quit`.
`/stats` показывает TTFT, токены, ток/с, стоимость и причину остановки
(`eos` / `length` / `repeat`).

Флаги, которые чаще нужны: `--max-tokens` (по умолчанию 256), `--temperature`
(у ВТМ — через `VTM_TEMPERATURE`, `VTM_TOP_P`, `VTM_REP_PENALTY`), `--ctx`,
`--no-cache-prefix` (отключить кэш префикса — он даёт до 13× на повторном вопросе).

### Какие модели годятся для чата

| Модель | Для чата | Почему |
|---|---|---|
| `runs/q4/3b-q4k.vtm` | ✅ | Q4_K нативно, качество как у ollama 3B |
| `runs/q4/3b-q4_1.vtm` | ✅ | Q4_1 с минимумами, чуть хуже и больше файлом |
| `runs/tiny/model.vtm` | ⚠️ | обученная, но 4M параметров |
| `demo/model.vtm` | ❌ | эмбеддинг (vocab 128), в чате скрыта |
| холодный экспорт (`runs/cold/*`) | ❌ | текст нечитаем — см. вердикт по тритам |

Токенизатор ВТМ: `tokenizer.json` + `tokenizer_config.json` рядом с моделью
(в `tokenizer_config.json` лежит EOS — без него генерация не останавливалась).
Путь можно задать `EVA_VTM_TOKENIZER`.

### Голос: как включить озвучку (Silero)

По умолчанию чат молчит. Основной локальный голос — **Silero** (русская модель
`v4_ru`, ставится один раз, нужна сеть), `say` остаётся аварийным, ElevenLabs
включается ключом:

```bash
sh tools/install_silero.sh         # torch в vendor/torch-py + модель v4_ru
```

`install_silero.sh` ставит torch для python3.13 (~780 МБ), качает модель
v4_ru (~40 МБ) в `vendor/silero`, затем сам синтезирует «Проверка голоса» и
печатает время/размеры. После установки сервер подхватывает голос сам:

```bash
python3 tools/chat_server.py --port 8765   # авто: ElevenLabs (ключ) → Silero → say
# принудительно: EVA_TTS=silero|system|null, голос: SILERO_SPEAKER=kseniya
# облако: ELEVENLABS_API_KEY=… (сбой/квота → автоматический откат на Silero/say)
```

Порядок автовыбора: **ElevenLabs** (если задан `ELEVENLABS_API_KEY`) → **Silero**
(если установлен) → **`say`** (macOS) / `espeak` (Linux) → null. При ключе
ElevenLabs выбирается первым; `EVA_TTS=silero` оставляет локальный голос
принудительно ([0197](ai/tasks/0197-switch-to-silero-only.md)).

Голос выбирается в интерфейсе (селектор «голос» на странице чата и в
приложении) или переменной `SILERO_SPEAKER`:

| Голос | Тембр |
|---|---|
| `baya` | женский, живой (по умолчанию) |
| `kseniya` | женский, чуть выше |
| `xenia` | женский, ниже |
| `aidar` | мужской (в селекторах скрыт, доступен через API) |

В интерфейсе появится галочка «озвучивать»: ответ режется на фразы, первая
звучит, пока генерируются остальные. Перед синтезом текст нормализуется:
числа, проценты, даты, телефоны, email, знаки и сокращения проговариваются
словами ([0205](ai/tasks/0205-tts-numbers.md),
[0206](ai/tasks/0206-tts-symbols.md)), а Ева говорит о себе в женском роде
([0207](ai/tasks/0207-feminine-speech.md)). Приватность: после установки
Silero синтез идёт **полностью локально**, текст никуда не уходит; сеть нужна
установщику, а ElevenLabs — только если задан ключ. Без Silero, ElevenLabs и
`say` галочка покажет «голос недоступен», и чат продолжит работать текстом.
Почему обёртка: `torch` ставится Python-модулем (колёса есть для python3.13,
для 3.14 их нет), а `eva/tts.py` держит его в отдельном процессе и не тащит
~300 МБ в память чат-сервера.

Silero синтезирует **долгоживущий worker** (`tools/silero_worker.py`): модель
`v4_ru` грузится один раз при первой фразе, дальше фразы идут по JSON-протоколу
через пайп. На M1 Pro: первая фраза 2,8–3,3 с (из них ~1,6–1,8 с — старт
worker'а), каждая следующая **0,04–0,18 с**, повтор из кэша ~0,2 мс; один
процесс на чат, 0,84–0,93 ГБ RSS, idle CPU ~0 % ([0193](ai/tasks/0193-silero-tts.md),
[0195](ai/tasks/0195-silero-worker.md)). Если первая фраза должна быть
быстрой, поднимите worker заранее (`prewarm_silero_worker()` в коде сервера —
после prewarm первая фраза идёт 0,04–0,18 с). Падение или таймаут worker'а
(20 с) — откат на системный `say`, чат не падает. Замеры и детали —
[docs/hybrid.md](docs/hybrid.md).

### Плагины

Расширения Евы — каталоги `plugins/<id>/` с манифестом `plugin.json` и
способностями `skill.fetch`/`filter.run`/`tts.engine`/`command`; ядро не знает
про конкретные данные ([ADR 0022](docs/adr/0022-plugins.md)). Рантайм готов
([0209](ai/tasks/0209-plugin-runtime.md)), включённые плагины перечислены в
`config/plugins.json` и переключаются на лету через `POST /api/plugins`:
`time`, `currency`, `stocks` (факты в системный блок), `fairy-tales` и
`music` (отвечают сами, без модели), `ru-numbers`, `ru-gender` и `ru-facts`
(фильтры речи и ответа). Что это значит на практике — какая фраза что делает —
**[docs/commands.md](docs/commands.md)**; контракт и как добавить свой плагин —
[docs/plugins.md](docs/plugins.md).

### Если что-то не так

| Симптом | Причина и что делать |
|---|---|
| `token=… вне vocab=128` | выбрана демо-модель; возьмите `runs/q4/…` |
| Ответ повторяет фразу | слабое квантование; включён страж повторов, помогает компактный промпт |
| Ответ длинный и «не по делу» | модель не удержала 6 правил — используйте `system.compact.md` |
| `ollama не отвечает` | `ollama serve` в отдельном терминале |
| Первый ответ 20 с, дальше 1 с | так и должно быть: загрузка + кэш префикса |
| «голос недоступен» | не установлен Silero; запустите `sh tools/install_silero.sh` |

## Гибрид

Ассистент — не только веса: скиллы отвечают из каталога (0 ₽, без галлюцинаций), малая
модель — тон и простое, большая — то, где малой не хватает знаний. Маршрут выбирается
правилами, у каждого ответа есть цена.

- `docs/hybrid.md` — контракт модулей: типы, маршрутизатор, сервер, телеметрия.
- `docs/adr/0009-hybrid-router.md` — почему гибрид, а не своя большая модель.
- `docs/hybrid-runbook.md` — запуск, `.env`, подключение из orioncode, метрики и 429.

