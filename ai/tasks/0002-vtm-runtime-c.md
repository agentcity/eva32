---
task_id: "0002"
type: feature
title: Виртуальная троичная машина: C-интерпретатор
created_at: 2026-11-18T03:00:00Z
author: yuri
assignee: agent:vtm-c
status: done
scope:
  docs: false
  code: true
  data: false
---

# Виртуальная троичная машина: C-интерпретатор

## Проблема

bitnet.cpp не даёт контроля: диктует формат весов, прячет ядра, тянет llama.cpp.
Нужен свой рантайм, который исполняет наш формат тритов и позже переедет на другое железо.

## Требования

- `src/vtm/`: загрузчик `.vtm` по `docs/vtm.md`, цикл инструкций, скалярные ядра.
- Операции v0.1: TMV, MM, RMSN, ROPE, SM, SILU, MUL, ADD, EMB, KVW, ATTN, SAMP, CPY, SCAL, HALT.
- Кривой файл — внятная ошибка, не segfault (границы offset+size, магия, версия).
- `make test`: разбор, отказ на битом файле, TMV на посчитанной вручную матрице,
  трит-кодек на контрольных векторах из `docs/trit-format.md`.
- CLI `vtm run model.vtm --tokens ... --max-new N`.

## Критерии приёмки

[ ] `cd src/vtm && make && make test` — зелёно
[ ] Обрезанный файл возвращает ошибку, а не падает
[ ] Отчёт: что быстрее — LUT243 или распаковка, с числом
[ ] Семантика ATTN/ROPE/SM описана в коде так, чтобы Python-эталон мог повторить

## Контекст и ограничения

Зона: только `src/vtm/` и `tests/vtm/`. Коммиты не делать — сдавать правки.
Нет внешних зависимостей, C99, `-O2 -Wall -Wextra -Werror`.

## Отчёт исполнителя

Сделано: `src/vtm/` — загрузчик `.vtm` с проверками границ, цикл интерпретатора,
скалярные ядра (LUT243 и direct), кодек тритов, CLI (`run`, `inspect`, `bench`),
`tests/vtm/test_vtm.c`.

Проверено архитектором при приёмке:
- `make clean && make && make test` → «все тесты VTM: ok», без warning под `-Werror`
- битый файл: exit 1 с сообщением, без падения; ASan+UBSan чисто, leaks 0
- **сверка с Python-эталоном**: на `tools/vtm_compile.py --synthetic` (attn и mlp,
  128 логитов) max|Δ| = 4,99e-10, argmax совпал (48 и 66) — замер мой, не из отчёта
- LUT243 быстрее прямой распаковки в 7,58× на 512×512 (block 160) и 3,50× на
  1024×1024 (block 128)

Ограничения: meta JSON не читается (rope_theta жёстко 10000 — для Qwen нужен 1e6),
SAMP только greedy, нет mmap и потоков, API не потокобезопасен.

- Спека: `docs/vtm.md`
- Формат: `docs/trit-format.md`
