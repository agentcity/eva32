---
task_id: "0048"
type: feature
title: Стриминг в роутере гибрида
created_at: 2026-11-18T08:00:00Z
author: yuri
assignee: agent:eva-stream-router
status: done
scope:
  docs: false
  code: true
  data: false
---

# Стриминг в роутере гибрида

## Проблема

SSE на сервере есть, но роутер умеет только «ответ целиком»: стримить нечего.
Для голоса и для живого чата нужен поток токенов от большой модели.

## Требования

- `router.stream(request, config, deps)` — генератор чанков: сначала маршрут
  (skill отвечает одним куском), затем поток от клиента, затем финальный чанк с
  route/cost. Скиллы стримят одним чанком (они детерминированы).
- Клиент: `stream_chat()` с разбором SSE от провайдера, таймауты, отмена.
- Сервер использует `router.stream`, формат событий не менять (уже есть SSE).
- Ошибки: если поток оборвался — финальное событие с `error`, без падения сервера.

## Критерии приёмки

[ ] `python3 -m unittest tests.test_router_stream -v` зелёный
[ ] сервер на стриме отдаёт те же события, что раньше (тесты не сломаны)
[ ] скилл-вопрос отдаётся одним чанком с route и cost 0
[ ] обрыв потока не роняет сервер (тест с заглушкой)
[ ] `python3 -m unittest discover -s tests` зелёный

## Контекст и ограничения

- Зона: eva/router.py, eva/clients/openai_compat.py, eva/server.py, tests/test_router_stream.py
- Другие файлы не трогать: параллельно работают агенты по `src/vtm/` (0050/0051/0053).
- Только stdlib; сеть запрещена в тестах.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

### Что сделано

- `eva/router.py` — добавлен `stream(req, cfg, deps) -> Iterator[StreamChunk]`
  и тип `StreamChunk` (`text`, `route`, `skill`, `usage`, `cost_rub`,
  `latency_ms`, `done`).
- `eva/clients/openai_compat.py` — добавлен `OpenAIClient.stream_chat()`
  (разбор SSE, таймауты, `Retry-After`, повтор только до первого куска,
  закрытие ответа в `finally`), транспорт `_open_stream` / `StreamResponse` /
  `StreamChunk` клиента; `ClientError` получил `kind="stream"` и
  `retry_after`. Если задан только `transport` от `chat()`, его буферизованное
  тело тоже разбирается как SSE — тестам и старым подменам достаточно одного
  сечения.
- `eva/server.py` — боевая связка `create_server(router=None)` теперь стримит
  через `default_stream_router` → `eva.router.stream`; итератор роутера
  закрывается в `finally` (в т.ч. при уходе клиента). Формат SSE-событий не
  менялся.
- `tests/test_router_stream.py` — новый файл, 31 тест: контракт, скилл,
  поток модели, cost, фолбэки, бюджет, SSE-клиент, интеграция с сервером.

### Контракт потока

`router.stream(request, config, deps)` — генератор:

1. решение о маршруте (`_decide`, те же правила, что у `answer`);
2. `skill` — **один** чанк: текст целиком + `route`, `skill`, `usage` с нулями,
   `cost_rub=0.0`, `done=True`;
3. `small`/`big` — куски `StreamChunk(text=..., done=False)` по мере генерации
   от `client.stream_chat()`, затем финальный чанк `text=""`, `done=True` с
   `route`, `usage`, `cost_rub`, `latency_ms`;
4. ошибка до первого куска — те же фолбэки, что в `answer` (small→big,
   big→small на 429/бюджет); ошибка после первого куска пробрасывается —
   сервер отдаёт событие `error` (`code=stream_error`) и `[DONE]`, не падая.

`client.stream_chat()` отдаёт `StreamChunk` клиента: текстовые дельты, затем
терминальный кусок с `usage` (или `{}`, если провайдер молчит). Ошибки —
только `ClientError`: `timeout`, `network`, `http`, `response`, `stream`.
Для клиентов без `stream_chat` роутер деградирует до одного куска из `chat()`.

### Как считается cost в потоке

1. Провайдер прислал `usage` (в т.ч. `cost_rub`) — берём его; цена провайдера
   важнее локального тарифа.
2. Прислал токены без цены — считаем по `cfg.estimate_cost(kind, prompt,
   completion)`.
3. Не прислал ничего — оцениваем токены по длине prompt и уже
   сгенерированного текста (`policy.estimate_tokens`, ~4 символа/токен) и
   считаем по тарифу `cfg`; в `usage`/финальный чанк попадают эти оценки.
4. Итоговая цена списывается в `Budget` и попадает в `cost_rub` финального
   чанка; превышение фактической цены над бюджетом отмечается в
   `route.reason` так же, как в `answer`.

### Проверки

```
$ python3 -m unittest tests.test_router_stream -v
Ran 31 tests in 0.063s
OK

$ python3 -m unittest tests.test_server -v
Ran 70 tests in 0.270s
OK

$ python3 -m unittest discover -s tests
Ran 919 tests in 12.403s
OK (skipped=2)
```

Ручная проверка на заглушках:

```
SKILL: chunks=1 route=skill skill=pricing cost=0.0 done=True text='Старт — 490 ₽'
MODEL: chunks=5 text_chunks=4 text='Привет! Чем помочь?' final_route=small cost=0.0 done=True
```

Дополнительно проверен настоящий транспорт на loopback-SSE (не заглушка
`urlopen`):

```
LOOPBACK SSE: chunks=4 text='Привет!' usage={'prompt_tokens': 5, 'completion_tokens': 3,
 'total_tokens': 8, 'cost_rub': 0.001} done=True
```

И вся боевая цепочка `create_server(router=None)` → `default_stream_router` →
`router.stream` → `OpenAIClient.stream_chat` → SSE-заглушка модели на loopback:

```
DEFAULT SERVER + LOOPBACK MODEL: status=200 text='Привет!' route=small cost=0.002 done=True
```

### Риски и что осталось

- Провайдер обязан поддерживать `stream_options.include_usage`; если он
  отвергает незнакомое поле — это будет HTTP-ошибка (лечится конфигом/флагом,
  сейчас не выносилось). Без `usage` цена считается оценкой по длине текста.
- SSE-разбор считает каждую строку `data:` отдельным событием (OpenAI так и
  шлёт); многострочные `data:`-поля не поддерживаются.
- Повтор потока возможен только до первого куска: после текста ошибка
  отдаётся клиенту событием `error`, а не повторным запросом к провайдеру.
- `retry_after` для потока учитывается из `Retry-After`; HTTP-дата, как и в
  `chat()`, игнорируется в пользу backoff.
- Если инжектирован только `transport` (как в старых тестах `chat()`), поток
  читается из уже прочитанного тела: работает, но без инкрементальности.
  Для настоящего стриминга нужен `stream_transport` / `_open_stream`.
- Тесты полного прогона на этом дереве могут плавать из-за параллельных
  агентов (в момент работы менялись `qat/`, `src/vtm/`); финальный прогон
  зелёный.
