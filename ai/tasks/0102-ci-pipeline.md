---
task_id: "0102"
type: feature
title: CI: сборка, тесты, санитайзеры одной командой
created_at: 2026-11-18T16:20:00Z
author: yuri
assignee: agent:ci-pipeline
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# CI: сборка, тесты, санитайзеры одной командой

## Проблема

Всё проверяется вручную: `make test`, `python3 -m unittest`, ASan/UBSan прогоны агенты
делали скриптами в /tmp. Нет единой команды и нет CI: регрессии ловятся глазами.

## Требования

- `make ci` в `src/vtm`: обычная сборка + `make test` + UBSan-прогон (`-fsanitize=undefined`)
  + Python-набор (`python3 -m unittest discover -s tests`), с внятным итогом.
- `tools/ci.py` — то же на уровне репозитория (для GitHub Actions): C-тесты, Python-тесты,
  `tools/check_tasks.py`, `tools/check_vtm_docs.py`, проверка что ни один тест не требует
  собранных артефактов (уже дважды ловили красный клон).
- `.github/workflows/ci.yml`: ubuntu-latest, python 3.12, шаги выше. Разработка на macOS —
  добавить матрицу macos тоже, если не сильно замедлит.
- Проверка: чистый клон → `python3 tools/ci.py` → зелёный.

## Критерии приёмки

[ ] `make ci` проходит на чистом дереве (вывод в отчёте)
[ ] `python3 tools/ci.py` из чистого клона зелёный
[ ] workflow-файл валиден (синтаксис YAML), шаги описаны
[ ] UBSan-прогон включён и чистый

## Контекст и ограничения

- Зона: src/vtm/Makefile, .github/workflows/ci.yml (новый), tools/ci.py (новый)
- Другие файлы НЕ трогать: параллельно работают семь других агентов по своим зонам.
- **Запрещено:** pip, сеть (кроме случаев, где явно разрешено), запуск сервера на порту,
  git-коммиты, скачивание моделей.
- **Отчёт пиши первым блоком ответа** — агенты этой сессии падают на финальном шаге.
- Проверка: `python3 -m unittest discover -s tests` (сейчас 1365 тестов) и, где нужно,
  `cd src/vtm && make test`.

## Отчёт исполнителя

_Заполняется по завершении._
