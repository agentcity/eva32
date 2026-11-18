---
task_id: "0015"
type: feature
title: Гибрид — eva-skills
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-skills
status: done
scope:
  docs: false
  code: true
  data: true
---

# Гибрид: eva-skills

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

Скиллы — детерминированные ответы из каталога: цены, часы, возврат, контакты. Ноль рублей, ноль галлюцинаций.

## Требования

- `catalog.py`: загрузка `data/catalog.json`, поиск по имени и синонимам, грубый поиск по словам.
- Скиллы по протоколу из `docs/hybrid.md`: `match(text)` / `run(text, ctx)`, БЕЗ вызова модели.
- `registry.try_skills(text, ctx)` — первый совпавший скилл или None.
- `data/catalog.json`: 10–15 услуг/тарифов с ценами, лимитами, заметками и ссылками `[1]`.
- Ответ скилла содержит источники `[n]`, чтобы формат совпадал с моделью.

## Критерии приёмки

[ ] `python3 -m unittest tests.test_skills -v` зелёный
[ ] «сколько стоит тариф Старт» → ответ без модели, с ценой и `[1]`
[ ] «какие часы работы» и «вернуть деньги» попадают в свои скиллы
[ ] неизвестный вопрос → `try_skills` возвращает None (скилл молчит)

## Контекст и ограничения

- Зона: eva/catalog.py, eva/skills/__init__.py, eva/skills/base.py, eva/skills/registry.py, eva/skills/pricing.py, eva/skills/hours.py, eva/skills/refunds.py, eva/skills/contacts.py, data/catalog.json, tests/test_skills.py
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

Сделано: `eva/catalog.py`, `eva/skills/` (6 модулей), `data/catalog.json`
(13 позиций), 75 тестов.

Проверено архитектором на живом прогоне без подмен:
- «Сколько стоит тариф Старт?» → pricing, «990 ₽/мес [1]… до 3 сотрудников»
- «Какие часы работы?» → hours, «будни 9–19, суббота 10–16»
- «Как вернуть деньги?» → refunds, «14 дней с момента оплаты»
- «Какая погода в Москве?» → молчит (правильно, скилла нет)
- сквозной `answer()` без подмен: route=skill, skill=pricing, **cost_rub=0.0**

Пробел, найденный при приёмке: «Как с вами связаться?» не поймал `contacts` —
формулировка не совпала с триггерами. Отмечено для 0007 (замена каталога).

Инфраструктурное: `data/catalog.json` попадал под `.gitignore` (`data/`), то есть
на чистом checkout скиллы падали. Правило исправлено: `data/*` + `!data/catalog.json`.

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
