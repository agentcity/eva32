---
task_id: "0200"
type: bugfix
title: Чат-шаблон Qwen и бюджет размышлений — 30B перестаёт «думать вечно»
created_at: 2026-11-26T10:00:00Z
author: yuri
assignee: agent:qwen-template
status: in_progress
scope: {docs: true, code: true, data: false}
---

# Сначала шаблон, потом бюджет

## Две проблемы (обе про большую модель, Qwen3-30B-A3B)
1. **Много времени уходит на `<think>`**: reasoning-токены идут по 1,9 ток/с —
   300 токенов «размышлений» это 2,5 минуты до первого слова ответа.
2. **Ответ не появляется вовсе**: размышления съедают весь `max_tokens`,
   и до ответа дело не доходит.

## Корень (найдено в коде)
`eva/backends/vtm_backend.py::format_chat_prompt` собирает промпт как
`role: content` + `assistant:` — это **не** шаблон Qwen. Qwen3 ждёт
`<|im_start|>role\n…<|im_end|>` и умеет отключать рассуждения
(`enable_thinking=false` в шаблоне / суффикс `/no_think`).

## Требования
1. **Настоящий шаблон Qwen** `eva/chat_template.py`: рендер
   `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`,
   с флагом `enable_thinking`; определять Qwen по `tokenizer_config.json`
   (есть `chat_template`) и использовать его; для не-Qwen — прежнее поведение.
2. **Рассуждения выключены по умолчанию** в чате (`--think` включает обратно):
   тогда удаляются и задержка, и обрыв ответа.
3. **Бюджет, если `--think`**: `--reasoning-budget N` (default 256). При
   исчерпании — принудительно закрыть рассуждение и **досказать ответ**:
   второй вызов с промптом `…assistant\n<think>…рассуждение…<｜end▁of▁thinking｜>\n` и
   `max_tokens = answer_budget`. Ответ не теряется ни при каком N.
4. `max_tokens` везде считать как `reasoning_budget + answer_tokens`, а не
   единым лимитом (API/чат/приложение).
5. Замеры: 30B — TTFT и tok/s с `--think` и без, на одном промпте; проверить,
   что при `--think --reasoning-budget 64` ответ появляется.

## Критерии приёмки
- [ ] без `--think` 30B отвечает **без** блока размышлений и быстрее (числа)
- [ ] с `--think --reasoning-budget 64` ответ досказывается, а не обрывается
- [ ] Qwen-шаблон применяется автоматически (проверить на 3B и 30B)
- [ ] тесты: рендер шаблона, бюджет, «ответ не теряется»
- [ ] полный набор зелёный

## Отчёт исполнителя

**Статус: сделано.** Шаблон Qwen применяется автоматически, рассуждения по
умолчанию выключены, при `--think` бюджет не теряет ответ: на 30B с
`--reasoning-budget 64` ответ появился и был досказан до конца.

### Что изменено

| Файл | Что |
|---|---|
| `eva/chat_template.py` (новый) | ChatML-рендер `<|im_start|>role\n…<|im_end|>\n` + `assistant`; `enable_thinking` (False — пустой закрытый `<think>\n\n</think>\n\n`, True — модель открывает `<think>` сама); определение по `tokenizer_config.json` рядом с токенизатором (для `qwen3-tokenizer.json` — `qwen3-tokenizer_config.json`); опция `/no_think`-суффикса; `reasoning_steps` — бюджет + досказывание; `strip_think` |
| `eva/backends/vtm_backend.py` | `VtmBackend(think, reasoning_budget, chat_template, no_think_suffix)`; `prompt_builder` по умолчанию — Qwen-рендер, если конфиг говорит Qwen, иначе прежний `format_chat_prompt`; `render_chat()`; `chat`/`stream_chat` считают `max_tokens` лимитом ответа, бюджет — сверху; `_native_token_limit` (потолок C-контекста = бюджет + ответ) |
| `tools/chat.py` | `--think` / `--reasoning-budget N` (default 256); `VtmChatBackend` рендерит шаблон с рассуждением и при исчерпании бюджета закрывает `</think>\n` и **досказывает ответ вторым прогоном**; `ChatSession.reasoning_budget` резервирует место в контексте; в историю пишется ответ без `<think>`; `--stats` показывает токены размышления; `--think` без Qwen3-токенизатора — понятная ошибка конфига (код 2) |
| `tools/chat_server.py` | `--think` / `--reasoning-budget` (флаги сервера, по умолчанию выключено), проброс в фабрику `build_backend(..., think=…)`; API-контракт `POST /api/chat` не менялся (`max_tokens` — лимит ответа), EVA.app править не понадобилось |
| `tests/test_chat_template.py` (новый) | 37 тестов: рендер, детект (Qwen3/Qwen2.5/не-Qwen), бюджет на моке, «ответ не теряется», по умолчанию без `<think>`, потолок C-контекста, проброс флагов сервера |
| `docs/vtm.md`, `docs/hybrid.md` | разделы про шаблон, бюджет и замеры |

`src/vtm/*`, `eva/tts.py`, `apps/eva/*` не трогал.

### Замеры (30B, один прогон на режим)

`runs/q4/qwen3-30b-a3b.vtm`, `runs/q4/qwen3-tokenizer.json`,
`VTM_Q4K_CHECK_OFF=1`, 16 ГиБ RAM, load average 7–12 (машина общая).
Промпт один и тот же: `What is the capital of France?`.

```bash
VTM_Q4K_CHECK_OFF=1 python3 tools/chat.py --backend vtm \
  --model runs/q4/qwen3-30b-a3b.vtm --tokenizer runs/q4/qwen3-tokenizer.json \
  --once "What is the capital of France?" --max-tokens 32 --stats
# → The capital of France is Paris.
#   [7 ток · 6.5 ток/с · TTFT 7450 мс · 31 симв · 0.0000 ₽]

VTM_Q4K_CHECK_OFF=1 python3 tools/chat.py --backend vtm \
  --model runs/q4/qwen3-30b-a3b.vtm --tokenizer runs/q4/qwen3-tokenizer.json \
  --think --reasoning-budget 64 --max-tokens 32 --stats \
  --once "What is the capital of France?"
# → <think>Okay, the user is asking…</think>
#   The capital of France is **Paris**. …
#   [64 ток (32 размышление) · 2.0 ток/с · TTFT 6485 мс · 276 симв · 0.0000 ₽]
```

Второй прогон (после исправления потолка C-контекста, см. ниже) — с
поминутными метками (разовый скрипт-стенд, в репозиторий не входит;
`--max-tokens 64` — лимит ответа, `--reasoning-budget 64`):

| режим | промпт | ответ | TTFT | первое слово ответа | декод ответа | всего |
|---|---|---|---|---|---|---|
| без `--think` | 19 ток | 7 ток | 10,5 с | 10,5 с | 4,0 ток/с | 12,3 с |
| `--think --reasoning-budget 64` | 15 ток | 64 размышл. + 53 | 8,6 с | **55,8 с** | 2,08 ток/с | 81,2 с |

Те же числа через `tools/chat.py --stats` (первый прогон, другая загрузка
хоста): без think — TTFT 7,45 с, 6,5 ток/с; с think — TTFT 6,49 с (до первого
токена рассуждения), 2,0 ток/с. Разброс — соседняя нагрузка, как и
предупреждает `docs/vtm.md`.

3B (`runs/q4/3b-q4k.vtm`, `runs/q4/tokenizer.json`): ответ
«The capital of France is Paris.», 7 ток, TTFT 13,3 с (включая загрузку
C-контекста), 33 ток/с. ChatML применяется, `<think>` нет (Qwen2.5 не
рассуждает).

### «Ответ не теряется»

* `--think --reasoning-budget 64`, `--max-tokens 32`: в первом замере
  рассуждение упёрлось не в бюджет, а в **потолок C-контекста**
  (`VtmBackend(max_tokens=32)` → C обрезал первый прогон на 32 токенах,
  бюджет 64 молча стал 32). Исправлено `_native_token_limit`: потолок =
  бюджет + ответ; после правки `usage.reasoning_tokens=64`, `answer_tokens=53`,
  `finish=stop`, ответ полный.
* Если модель успела сама закрыть `</think>` — второго закрытия нет, второй
  прогон продолжает с этого места; если первый прогон закончился EOS —
  ответ уже целиком сгенерирован, второго прогона нет (юнит-тесты на моке).
* `--max-tokens` теперь везде лимит **ответа**; полный лимит новых токенов =
  `reasoning_budget + max_tokens` (чат, API, потолок C-контекста, резерв
  контекста в `ChatSession`).
* Страж зацикливания/стоп-маркеров к рассуждению не применяется — иначе
  обрыв «мысли» съедал бы ответ; рассуждение и так ограничено бюджетом.

### Проверка

```bash
python3 -m unittest tests.test_chat_template -v     # 37 ok
python3 -m unittest discover -s tests -q            # 2570 tests, OK (skipped=71)
python3 tools/check_docs_dates.py --today 2026-11-26 # даты и 727 ссылок в порядке
```

### Риски и ограничения

* **Рассуждение не стало дешёвым.** Бюджет гарантирует ответ, а не скорость:
  на 30B 64 токена `<think>` — это 37 с, плюс 18,6 с на prefill второго
  прогона (кэш префикса хранит состояние после промпта, а не после
  генерации), плюс сам ответ. Поэтому по умолчанию выключено; включать
  осознанно.
* **Абсолютные секунды зависят от загрузки хоста** (load 7–12, соседние
  агенты): TTFT на 30B гулял 7,5–10,5 с в одном и том же коде. Замеры по
  одному прогону на режим, как и просили; парных «до/после» нет.
* **`/no_think`-суффикс реализован, но по умолчанию не включён**: пустого
  закрытого `<think>` хватило — на 30B блок размышлений не появился (ответ
  сразу). Опция оставлена для моделей, где шаблон не сработает.
* **Детект по имени файла**: `tokenizer.json` → `tokenizer_config.json`,
  `qwen3-tokenizer.json` → `qwen3-tokenizer_config.json`. Если конфиг назван
  иначе, шаблон не найдётся и останется прежний `role: content` (для 30B это
  означало бы «нет рассуждений и другой промпт»). Для 30B нужно явно
  передавать `--tokenizer runs/q4/qwen3-tokenizer.json`; в `tokenizer.json`
  из `runs/q4` (Qwen2.5) токенов `<think>`/`</think>` нет.
* **Детект требует ChatML-маркеров**, а не просто наличия `chat_template`:
  шаблон без `<|im_start|>` (Llama и подобные) не применяется — остаётся
  прежний `role: content`. Это защита от чужого формата; для Qwen и Qwen2.5/
  Qwen3 маркеры есть.
* **Tool-роли**: ChatML-рендер выводит роль как есть; формат `<tool_call>`
  Qwen3 не реализован — в текущих путях (router/чат) tool-сообщений нет.
* **API**: `think` — флаг сервера, не поле запроса; per-request переключение
  и EVA.app не добавлялись. `POST /api/chat` не менялся, `max_tokens` в нём
  теперь лимит ответа (при выключенном `--think` поведение прежнее).
* **Ollama/OpenAI**: `--think`/`--reasoning-budget` действуют только на ВТМ;
  для других бэкендов CLI предупреждает и игнорирует флаг.
* **История**: в `chat.py`/`chat_server.py` рассуждение в историю не пишется
  (как и раньше в сервере); если бюджет оборвал рассуждение до ответа, в
  историю попадает хотя бы оно, чтобы ход не терялся.
