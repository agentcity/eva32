---
task_id: "0017"
type: feature
title: Гибрид — eva-server
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-server
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Гибрид: eva-server

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

OpenAI-совместимый HTTP-сервер на stdlib: чтобы orioncode подключился как к обычному провайдеру.

## Требования

- `POST /v1/chat/completions` — формат OpenAI + дополнительное поле `eva` (маршрут, скилл, цена).
- `GET /health` — статус и версия.
- Сервер на `http.server`, без зависимостей; маршрутизатор инжектируется (в тестах — заглушка).
- Ошибки: 400 на битый JSON, 500 на исключение с человеческим текстом, не трейсбек.

## Критерии приёмки

[ ] `python3 -m unittest tests.test_server -v` зелёный
[ ] сервер реально стартует на свободном порту и отвечает 200 на /health
[ ] ответ /v1/chat/completions содержит `choices[0].message.content` и поле `eva`
[ ] битый JSON → 400, без падения процесса

## Контекст и ограничения

- Зона: eva/server.py, eva/__main__.py, tests/test_server.py
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
