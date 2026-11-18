---
task_id: "0016"
type: feature
title: Гибрид — eva-clients
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-clients
status: in_progress
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

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
