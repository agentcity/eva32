---
task_id: "0010"
type: infra
title: CI: тесты Python и C на каждый пуш
created_at: 2026-11-18T03:20:00Z
author: yuri
assignee: agent:ci
status: done
scope:
  docs: false
  code: true
  data: false
---

# CI: тесты Python и C на каждый пуш

## Проблема

Над репозиторием работают до восьми агентов одновременно. Без автопроверки
чужая правка ломает тесты молча, и это выясняется при приёмке, а не при пуше.

## Требования

- GitHub Actions: на push и pull_request — Python-тесты (`unittest discover`),
  сборка и тесты C (`src/vtm`), проверка JSON-реестра `ai/tasks/index.json`.
- Отдельная джоба проверяет, что все файлы задач имеют фронтматтер и критерии приёмки.
- `make ci` — тот же набор локально, одной командой, без Docker.
- Кэш зависимостей не нужен: у нас только stdlib и C.

## Критерии приёмки

- [x] `.github/workflows/ci.yml` валиден по YAML и запускает три джобы
- [x] `make ci` локально проходит на текущем HEAD
- [x] Проверка реестра задач падает, если файл задачи без `task_id` (проверено на копии)
- [x] В отчёте — вывод локального `make ci`

## Контекст и ограничения

- Зона: `.github/workflows/`, `Makefile` (только цель `ci` и её зависимости), `tools/check_tasks.py`.
- Не менять существующие цели Makefile и не трогать `qat/`, `distill/`, `src/vtm/`.

## Отчёт исполнителя

Сделано: `.github/workflows/ci.yml` (три джобы: python-тесты, C-тесты с мягким
пропуском при отсутствии `src/vtm/Makefile`, валидация реестра), `tools/check_tasks.py`
(stdlib, разбор фронтматтера построчно, 245 строк) и цель `make ci` в Makefile.

Проверено архитектором при приёмке:
- `make ci` → exit 0, «Ran 60 tests OK», C-шаг честно сообщает о пропуске, реестр OK
- негативный тест на копии: убрали `task_id` → `✗ задача 0010: нет task_id`, exit 1
- восстановление реестра → снова exit 0
- YAML провалидирован Ruby Psych (PyYAML в системе нет), джобы: python, c, tasks

Осталось: реальный C-шаг включится, когда задача 0002 положит `src/vtm/Makefile`
с целью `test`. Workflow на GitHub ещё не запускался — только локальный аналог.

## Ссылки

- Реестр: `ai/tasks/index.json`
- Правила: `ai/README.md`
