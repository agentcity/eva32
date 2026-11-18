---
task_id: "0034"
type: fix
title: Вход eva.cli, который обещают документация и runbook
created_at: 2026-11-18T05:35:00Z
author: yuri
assignee: saron
status: done
scope:
  docs: false
  code: true
  data: false
---

# eva.cli

## Проблема

`docs/hybrid.md` и рунбук обещают `python3 -m eva.cli ask "..." --explain` и
`python3 -m eva.cli serve`, но модуля нет — фактический вход только `python3 -m eva`.
Задача 0033 нашла расхождение при сверке документации с кодом.

## Требования

- `eva/cli.py`: подкоманды `ask` (текст, `--explain`, `--json`, `--user-id`) и
  `serve` (`--host`, `--port`), только stdlib.
- Отсутствие модели — человеческое сообщение и код 2, без трейсбека.
- Тесты: парсер, ответ через скилл без модели, ошибка без модели.

## Критерии приёмки

- [x] `python3 -m eva.cli ask "Сколько стоит тариф Старт?" --explain` отвечает скиллом, цена 0
- [x] общий вопрос без модели даёт сообщение, а не трейсбек (exit 2)
- [x] `python3 -m eva.cli serve --port 8010` поднимает тот же сервер, что `python3 -m eva`
- [x] `tests/test_cli.py` зелёный, полный набор не сломан

## Контекст и ограничения

- Зона: `eva/cli.py`, `tests/test_cli.py`.
- Сервер и роутер не дублируются — CLI только вызывает существующие функции.

## Отчёт исполнителя

Написан `eva/cli.py` (делегирует в `router.answer` и `server.create_server`),
5 тестов. Проверено: скилл отвечает за 10 мс с ценой 0; общий вопрос без модели
даёт «не могу ответить» и exit 2 без трейсбека.

## Ссылки

- `docs/hybrid.md`, `docs/hybrid-runbook.md`
