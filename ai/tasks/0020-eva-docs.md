---
task_id: "0020"
type: feature
title: Гибрид — eva-docs
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-docs
status: in_progress
scope:
  docs: true
  code: true
  data: false
---

# Гибрид: eva-docs

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

Рунбук гибрида: как запустить, настроить .env, подключить orioncode, читать метрики.

## Требования

- `docs/hybrid-runbook.md`: запуск сервера, переменные окружения, подключение из orioncode
  как OpenAI-провайдера, как смотреть маршруты и стоимость, что делать при 429.
- README: короткий раздел со ссылками на docs/hybrid.md и ADR 0009.
- Команды должны быть проверяемыми: не выдумывать флаги, которых нет.

## Критерии приёмки

[ ] `docs/hybrid-runbook.md` 80–140 строк, все bash-блоки проходят `bash -n`
[ ] команды соответствуют реальным CLI (`python3 -m eva.cli ...`)
[ ] README дополнен, ADR 0009 в таблице ADR

## Контекст и ограничения

- Зона: docs/hybrid-runbook.md, README.md (только раздел про гибрид)
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
