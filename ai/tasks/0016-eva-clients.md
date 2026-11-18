---
task_id: "0016"
type: feature
title: Гибрид — eva-clients
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-clients
status: done
scope:
  docs: false
  code: true
  data: false
---

# Гибрид: eva-clients

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

HTTP-клиент к OpenAI-совместимым моделям: малой (локальной) и большой (облачной). Ретраи, таймауты, учёт цены.

## Требования

- `OpenAIClient(base, key, model, timeout, retries)` с `.chat(messages, max_tokens)`.
- Повтор на таймаут/5xx/429 с backoff; после исчерпания — `ClientError` с понятным текстом.
- Возвращает text, usage (prompt/completion/cost_rub), raw.
- Сеть только через `urllib`; в тестах — подмена транспорта, никаких реальных запросов.

## Критерии приёмки

[ ] `python3 -m unittest tests.test_clients -v` зелёный, без сети
[ ] 500 и таймаут → ретраи, затем ClientError
[ ] 429 → ретрай с задержкой (в тесте задержка патчится)
[ ] cost_rub берётся из ответа, если провайдер его отдаёт

## Контекст и ограничения

- Зона: eva/clients/__init__.py, eva/clients/openai_compat.py, tests/test_clients.py
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

Сделано: `eva/clients/openai_compat.py` (668 строк), `__init__.py`, 55 тестов.

Проверено архитектором при приёмке (не по отчёту):
- 55 тестов клиента и весь набор (308) — OK
- 503 три раза → 3 попытки, паузы 0,5 и 1,0 с, ClientError с status=503
- ключ в тексте ошибки отсутствует (маскируется)
- 429 с `Retry-After: 2` → одна повторная попытка после паузы 2,0 с, затем успех
- транспорт подменяется: тест ставит «бомбу» на urlopen и доказывает, что сеть не зовётся

Риски, принятые осознанно: таймаут считается на попытку (худшая латентность
растёт с retries); провайдер без cost_rub даёт 0,0 → бюджет не увидит расход,
лечится прайсом в конфиге; ClientResult — Mapping, не dict (router уже делает dict()).

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
