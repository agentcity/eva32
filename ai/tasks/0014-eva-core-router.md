---
task_id: "0014"
type: feature
title: Гибрид — eva-core-router
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-core-router
status: done
scope:
  docs: false
  code: true
  data: false
---

# Гибрид: eva-core-router

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

Типы, конфиг и маршрутизатор: решают, кто отвечает на запрос — скилл, малая модель или большая.

## Требования

- `types.py`: Message, Request, Route, Answer — ровно как в `docs/hybrid.md`.
- `config.py`: конфиг из env (`EVA_SMALL_BASE/KEY/MODEL`, `EVA_BIG_*`, `EVA_MAX_COST_RUB`, пороги).
- `router.py`: `decide(req, cfg) -> Route` — скиллы → классификатор → small/big; `answer(req, cfg, deps) -> Answer`.
- `policy.py`: фолбэки (small→big при ошибке, big→small при 429/бюджете), таймауты, бюджет на запрос.
- Зависимости (skills, classifier, clients) подключаются лениво и подменяются в тестах.

## Критерии приёмки

[ ] `python3 -m unittest tests.test_router -v` зелёный без сети
[ ] `decide` возвращает reason для каждого маршрута
[ ] при ошибке малой модели и доступной большой — фолбэк с reason
[ ] при превышении бюджета — обрыв, не тихая трата

## Контекст и ограничения

- Зона: eva/__init__.py, eva/types.py, eva/config.py, eva/router.py, eva/policy.py, tests/test_router.py
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

Сделано: `eva/types.py`, `config.py`, `policy.py`, `router.py`, `__init__.py`, 50 тестов.

Проверено архитектором при приёмке (руками, не по отчёту):
- скилл отвечает → модели не вызываются вообще, cost_rub=0
- бюджет 0 при платной малой модели → BudgetExceeded (needed=0,618 ₽), ни один клиент не вызван
- малая падает → фолбэк на большую, cost берётся из usage
- конфиг иммутабельный (frozen dataclass) — правки только конструктором

Риски приняты: тарифы big 0.40/1.20 ₽ за 1k — консервативная оценка для pre-flight;
оценка completion = max_tokens может дать ложный downgrade; пороги латиницы пока
в classifier.py, не в Config.

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
