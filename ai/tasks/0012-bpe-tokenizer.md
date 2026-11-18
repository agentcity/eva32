---
task_id: "0012"
type: feature
title: BPE-токенизатор для ВТМ (C++17)
created_at: 2026-11-18T04:20:00Z
author: yuri
assignee: agent:tokenizer
status: done
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

- [x] `make test` в `src/vtm` собирает и прогоняет тесты токенизатора
- [x] round-trip: `decode(encode(x)) == x` для ASCII, кириллицы, эмодзи и смеси
- [x] спецтокен распознаётся и не разбивается BPE (есть тест)
- [x] битый `tokenizer.json` → сообщение и код возврата, без segfault
- [x] В отчёте — вывод тестов и время сборки

## Контекст и ограничения

- Зона: `src/vtm/tokenizer.*`, `src/vtm/test_tokenizer*`, `tests/vtm/fixtures/*.json`,
  `src/vtm/Makefile.tok`.
- `src/vtm/Makefile` НЕ трогать: там параллельно работает агент C-интерпретатора,
  два писателя затрут цели друг друга. Слияние целей делает архитектор при приёмке.
- Не трогать: интерпретатор, ядра, `docs/`, `distill/`, `qat/`, `ai/`.
- Реальный `tokenizer.json` Qwen в репозиторий не класть (объём); тесты — на фикстуре.

## Отчёт исполнителя

Сделано: `src/vtm/tokenizer.hpp/.cpp` (HF `tokenizer.json`: vocab, merges,
added_tokens, ByteLevel pre-tokenizer, декодер), `tok_main.cpp` (CLI `vtm-tok`
encode/decode/info), `tests/vtm/test_tokenizer.cpp`, фикстура с 287 токенами.

**Важно:** агент-исполнитель упал до финального отчёта, но оставил рабочий код.
Приёмку провёл архитектор лично:
- `make -f Makefile.tok test-tok` → «OK: 131 проверок, 0 провалов»
- round-trip кириллицы: `Привет, мир!` → 275,44,281,33 → `Привет, мир!`
- round-trip эмодзи: `Привет 👋 мир 🌍` → 10 токенов → тот же текст
- битый JSON: сообщение с позицией, без падения

Цели сборки слиты в общий `src/vtm/Makefile` (`make test-all`), как и планировалось.

## Ссылки

- `docs/vtm.md`
