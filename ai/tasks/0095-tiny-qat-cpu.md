---
task_id: "0095"
type: experiment
title: Крошечная модель: сквозная проверка цепочки на своём железе
created_at: 2026-11-18T13:45:00Z
author: yuri
assignee: agent:tiny-qat-cpu
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Крошечная модель: сквозная проверка цепочки на своём железе

## Проблема

Каждое звено проверено отдельно на синтетике, но **сквозь всю цепочку настоящие веса
не проходили ни разу**: QAT-обучение → экспорт `.trit` → компиляция `.vtm` →
C-интерпретатор → сверка с HuggingFace. Если на арендованном 1.5B что-то разойдётся,
искать причину придётся на железе за 1 890 ₽/сутки.

## Требования

Сделать `scripts/tiny_qat.py`: собрать крошечную Qwen2 (без скачивания весов),
обучить её QAT на наших данных и провести через всю цепочку.

Готовое окружение (уже установлено):
- `PYTHONPATH=.pylibs python3.13` — там torch 2.15 и transformers 5.23 (CPU);
- токенизатор: `tests/vtm/fixtures/tokenizer.json` (287 токенов),
  `PreTrainedTokenizerFast` из него; pad/eos задать явно;
- данные: `data/train.jsonl`, `data/val.jsonl` (3607/401, поля question/answer);
- QAT-код: `qat/model.py` (`apply_ternary(model, quant)`, `select_linear_modules`,
  `parameter_report`), `qat/config.py` (`QuantConfig`), `qat/checkpoint.py`,
  `qat/export.py`, `qat/train.py` (посмотри, что переиспользовать);
- компилятор: `tools/vtm_compile.py` (manifest → `.vtm`);
- C: `cd src/vtm && make` → `./vtm run model.vtm --tokens ... --max-new N`.

Конфигурация модели (подобрана, 4 009 984 параметра):
`Qwen2Config(vocab_size=287, hidden_size=256, intermediate_size=1024,
num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
max_position_embeddings=1024, tie_word_embeddings=True)`.

Шаги:
1. Токенизировать пары (question → answer), обрезать по 512 токенов.
2. Применить тернарные Linear через `qat.model.apply_ternary`.
3. Обучить 1–2 эпохи (AdamW, lr 3e-3 — для крошечной модели больше, чем 3e-5),
   логировать loss каждые 100 шагов; сохранить чекпоинт в `runs/tiny/`.
4. Экспорт (`qat/export.py` или напрямую) → `.trit` + `non_ternary.bin` + манифест.
5. `tools/vtm_compile.py` → `runs/tiny/model.vtm`.
6. Запустить C: `./src/vtm/vtm run runs/tiny/model.vtm --tokens <ids> --max-new 8`.
7. Сверить логиты C и HF на одном промпте: max|Δ| (порог 1e-4, объяснить, если больше).

## Критерии приёмки

- [ ] loss на val падает относительно старта (числа в отчёте)
- [ ] `.vtm` собран и C-интерпретатор его загружает
- [ ] C генерирует токены (пусть бессмысленные — цепочка важна)
- [ ] max|Δ| C↔HF на одном промпте измерен (число)
- [ ] `python3 -m unittest discover -s tests` зелёный

## Контекст и ограничения

- Зона: scripts/tiny_qat.py (новый), runs/tiny/ (артефакты), tests/test_tiny_qat.py (новый)
- НЕ трогать `qat/`, `src/vtm/`, `distill/`, `data/` — только читать.
- Сеть не нужна (веса не скачиваются); pip не нужен (всё стоит).
- Если обучение не сходится — сдай то, что есть, с числами и объяснением.
- **Отчёт первым блоком ответа.**
- Артефакты `runs/tiny/` в git не добавлять (gitignore).

## Отчёт исполнителя

_Заполняется по завершении._
