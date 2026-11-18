---
task_id: "0124"
type: feature
title: Golden компетенций отдельно от прайса
created_at: 2026-11-18T18:30:00Z
author: yuri
assignee: architect
status: done
scope:
  docs: true
  code: false
  data: true
---

# Golden — про умение

## Проблема

60% «золотого набора» было ценами и тарифами. Цена — не умение: она меняется, у каждого
бизнеса своя, и её место в RAG. Настоящее golden — рассуждение, обязательные уточняющие
вопросы, маркетинговые тексты, общение с клиентом, правила русского языка, инварианты.

## Что сделано

- `eval/golden_competence.jsonl` — 64 строки, 8 категорий × 8.
- `eval/golden_catalog.jsonl` — 45 строк прайса (регрессия RAG/скиллов).
- `tools/eval_competence.py` — структурные проверки + отчёт по категориям.
- `docs/golden.md` — зачем разделение и как пользоваться.
- `tools/eval_hybrid.py` переведён на каталог.

## Критерии приёмки

- [x] golden компетенций отделён от прайса
- [x] 8 категорий, проверки структурные и исполняемые
- [x] каталог проверяется отдельно (регрессия скиллов/RAG)
- [x] тесты и полный набор зелёные

## Отчёт исполнителя

Сделано: golden разнесён на два набора.

`eval/golden_competence.jsonl` — 64 строки, 8 категорий по 8: reasoning, clarify,
marketing, writing, client, language, behavior, limits. Проверки структурные
(asks_question, min/max_words, must_contain/not_contain, no_invented_prices,
forbid_numbers_from_catalog), поэтому набор работает как исполняемая спецификация
без модели.

`eval/golden_catalog.jsonl` — 45 строк прайса/каталога: это регрессия скиллов и RAG,
а не «умность». `tools/eval_hybrid.py` переведён на него (цена 1000 запросов
128,94 → 10,42 ₽, потому что почти всё закрывают скиллы; содержание 44/44).

Причина: цены меняются и у каждого бизнеса свои — модель должна брать число из контекста,
а не помнить наше. Файл `docs/golden.md` объясняет разделение и порядок использования.

Тесты: новый `tests/test_eval_competence.py` (12), полный набор 1653 OK.

## Ссылки

- `docs/golden.md`, `ai/tasks/0105-golden-classes.md`
