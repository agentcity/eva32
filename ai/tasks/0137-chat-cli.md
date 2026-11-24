---
task_id: "0137"
type: feature
title: Чат-адаптер: реплика с историей и стримингом для любых бэкендов
created_at: 2026-11-24T13:20:00Z
author: yuri
assignee: agent:chat-cli
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Чат-адаптер: реплика с историей и стримингом для любых бэкендов

## Проблема

Чтобы тестить модели, нужен один интерфейс: локальный `.vtm`, ollama,
OpenAI-совместимый API. Сейчас всё это только через curl/скрипты.

## Требования

- `tools/chat.py`: интерактивный REPL. `--backend vtm|ollama|openai`, `--model`,
  `--system`, `--ctx`, `--timeout`.
- История диалога (сообщения user/assistant), `/reset`, `/system`, `/quit`, `/stats`
  (токены/с, TTFT, потраченные ₽ для API).
- Стриминг по токенам; для vtm — `cache_prefix` (второй запрос ~1 с).
- Контекст и окно: обрезать историю по лимиту модели, не терять системный промпт.
- Тесты на моках бэкендов (без сети и без GPU).

## Критерии приёмки

[ ] `python3 tools/chat.py --backend ollama --model qwen2.5:3b` работает
[ ] `--backend vtm --model demo/model.vtm` работает, стриминг есть
[ ] `/stats` показывает TTFT и ток/с
[ ] тесты на моках, полный набор зелёный

## Контекст и ограничения

- Зона определяется названием задачи и файлами, которые в ней упомянуты.
- Параллельно работают пять других агентов — чужие файлы не трогать.
- **Запрещено:** pip, сеть (кроме ollama на localhost), git-коммиты, запуск
  сервера на внешнем интерфейсе, скачивание моделей HF без явного флага.
- **Отчёт пиши первым блоком ответа** — агенты этой сессии падают на финальном шаге.
- Проверка: `python3 -m unittest discover -s tests`; где нужно — `cd src/vtm && make test`.

## Отчёт исполнителя

Сделано (agent:chat-cli, 2026-11-24):

- **`tools/chat.py`** — REPL для трёх бэкендов: `--backend vtm|ollama|openai`,
  `--model`, `--system`, `--ctx`, `--timeout`, `--max-tokens`; дополнительно
  `--base`, `--key`, `--tokenizer`, `--price-prompt/--price-completion`,
  `--stats`, `--once`, `--no-cache-prefix`.
- **История** — user/assistant; команды `/reset`, `/system [текст|-|@файл]`,
  `/stats`, `/help`, `/quit`. Системный промпт хранится отдельно и не
  теряется ни при сбросе, ни при обрезке.
- **Обрезка** — бюджет `ctx - max_tokens - 8 - системный промпт`; старые
  сообщения выбрасываются целиком (текст вопроса не урезается молча), последний
  вопрос и системный промпт сохраняются, ведущий ответ ассистента без вопроса
  снимается. Факт обрезки печатается в stderr и виден в `/stats`.
- **Стриминг** — ВТМ: `VtmBackend.stream_ids` (реальный C-колбэк
  `vtm_api_generate` на каждый токен, декодирование чанками, точный счёт
  completion-токенов, EOS не считается); `cache_prefix` включён по умолчанию,
  `EVA_VTM_CACHE=0`/`--no-cache-prefix` выключают. ollama: свой
  `POST /api/chat` с `stream: true` (NDJSON, не SSE — так устроен ollama;
  парсер понимает и `data:`-строки). openai: SSE через
  `eva.clients.OpenAIClient` (ретраи, `usage`, цена).
- **`/stats`** — TTFT, prompt/completion/total токены, ток/с, символы
  (и симв/ток), время, причина остановки, стоимость (провайдер → тарифы →
  «— (тариф не задан)»), заполнение контекста, число обрезанных сообщений,
  точность счётчика токенов.
- **`tests/test_chat_cli.py`** — 83 теста на моках: история, обрезка,
  команды, REPL, парсер SSE/NDJSON, ошибки HTTP/потока, сборка бэкендов из
  env, CLI. Сети нет (тесты проходят при заблокированном `socket.socket`);
  GPU нет. Плюс один интеграционный тест настоящей ВТМ под `skipUnless`.

Проверено:

- `python3 -m unittest tests.test_chat_cli` → `Ran 83 tests ... OK`;
- те же тесты при `mock.patch("socket.socket", side_effect=...)` → 83 OK;
- `python3 tools/chat.py --help` → код 0;
- `--backend ollama --model qwen2.5:3b` (localhost) → ответ потоком,
  `/stats`: TTFT 7,7 с (первая загрузка модели), 42,1 ток/с, 20 токенов, 0 ₽;
  второй ход с историей: 77 prompt + 3 completion;
- openai-ветка против локального ollama `/v1/chat/completions` (настоящий
  SSE): «Париж.», 5 completion-токенов, цена по `--price-*` 0,0232 ₽;
- ВТМ: `runs/tiny/model.vtm` + `tests/vtm/fixtures/tokenizer.json` → поток
  по токенам, `/stats`: TTFT 105–185 мс, 8 токенов, 117 ток/с, 0 ₽;
- полный набор: `python3 -m unittest discover -s tests` → 1905 тестов, OK
  (35 скипов).

Ограничения:

- `--backend vtm --model demo/model.vtm`: команда запускается, но
  `demo/model.vtm` — демо-эмбендинг (`vocab=128`), текст он не генерирует.
  CLI ловит ошибку C-рантайма и печатает разбор с подсказкой взять текстовую
  модель; реальный стриминг проверен на `runs/tiny/model.vtm`.
- Стоимость для openai известна только если провайдер прислал `cost_rub`
  или заданы `--price-*`/`EVA_BIG_PRICE_*`; иначе печатается «тариф не задан».
- Реальный облачный OpenAI-совместимый провайдер не звался (запрет на сеть) —
  только моки и локальный ollama.

