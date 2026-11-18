---
task_id: "0012"
type: feature
title: BPE-токенизатор для ВТМ (C++17)
created_at: 2026-11-18T04:20:00Z
author: yuri
assignee: agent:tokenizer
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# BPE-токенизатор для ВТМ

## Проблема

Полная модель на ВТМ не может работать без токенизатора: интерпретатор умеет
исполнять инструкции, но текст в токены и обратно превращать некому. Сейчас у
нас есть только Python-сторона (HF), а рантайм должен быть самодостаточным.

## Требования

- `src/vtm/tokenizer.hpp` / `src/vtm/tokenizer.cpp` на C++17 без внешних зависимостей.
- Чтение `tokenizer.json` (формат HuggingFace, BPE): `model.vocab`, `model.merges`,
  `added_tokens`, `normalizer`, `pre_tokenizer` (ByteLevel), `decoder`.
- `encode(text)` → vector<int32_t>; `decode(ids)` → строка. Обработка UTF-8
  посимвольно с корректным разбиением многобайтовых символов.
- Byte-level отображение (GPT-2/Qwen-стиль): байт → unicode-символ и обратно.
- Спецтокены (`added_tokens`) находятся поиском подстроки, а не BPE.
- CLI: `vtm-tok encode "текст"`, `vtm-tok decode 1,2,3`, `vtm-tok info tokenizer.json`.
- Тесты `tests/vtm/test_tokenizer.cpp`: фикстура tokenizer.json в репозитории
  (маленький словарь с известными мержами), round-trip на ASCII, кириллице,
  эмодзи, спецтокене; падение с внятной ошибкой на битом JSON.

## Критерии приёмки

- [ ] `make test` в `src/vtm` собирает и прогоняет тесты токенизатора
- [ ] round-trip: `decode(encode(x)) == x` для ASCII, кириллицы, эмодзи и смеси
- [ ] спецтокен распознаётся и не разбивается BPE (есть тест)
- [ ] битый `tokenizer.json` → сообщение и код возврата, без segfault
- [ ] В отчёте — вывод тестов и время сборки

## Контекст и ограничения

- Зона: `src/vtm/tokenizer.*`, `src/vtm/test_tokenizer*`, `tests/vtm/fixtures/*.json`,
  `src/vtm/Makefile` (только добавление целей сборки теста; если файла нет — создай).
- Не трогать: интерпретатор, ядра, `docs/`, `distill/`, `qat/`, `ai/`.
- Реальный `tokenizer.json` Qwen в репозиторий не класть (объём); тесты — на фикстуре.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/vtm.md`
