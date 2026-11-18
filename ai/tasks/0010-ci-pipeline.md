---
task_id: "0010"
type: infra
title: CI: тесты Python и C на каждый пуш
created_at: 2026-11-18T03:20:00Z
author: yuri
assignee: agent:ci
status: in_progress
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

- [ ] `.github/workflows/ci.yml` валиден по YAML и запускает три джобы
- [ ] `make ci` локально проходит на текущем HEAD
- [ ] Проверка реестра задач падает, если файл задачи без `task_id` (проверить на копии)
- [ ] В отчёте — вывод локального `make ci`

## Контекст и ограничения

- Зона: `.github/workflows/`, `Makefile` (только цель `ci` и её зависимости), `tools/check_tasks.py`.
- Не менять существующие цели Makefile и не трогать `qat/`, `distill/`, `src/vtm/`.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- Реестр: `ai/tasks/index.json`
- Правила: `ai/README.md`
