---
task_id: "0126"
type: feature
title: Преflight аренды GPU — одна команда проверки и запуска
created_at: 2026-11-23T14:40:00Z
author: yuri
assignee: agent:preflight
status: in_progress
scope:
  docs: true
  code: true
  data: false
---

# Чтобы аренда не сгорела на настройке

## Проблема

Оплаченный час GPU дорог, а первый запуск на чужой машине упирается в мелочи:
версия CUDA, свободная память, отсутствие `tokenizer.json`, забытый `HF_TOKEN`,
не тот конфиг, отсутствие `runs/` и датасета. Каждая такая мелочь — минуты аренды.

## Требования

- `tools/rental_preflight.py`: проверяет окружение и печатает «можно/нельзя»:
  - `torch` и `torch.cuda.is_available()`, имя и объём VRAM;
  - свободная память против оценки из `docs/rental-runbook.md` (1.5B — 21,32 ГиБ);
  - наличие данных (`runs/mixture/train.jsonl` или `distill/out/*.jsonl`), их размер;
  - наличие `qat/configs/qwen1.5b.yaml` и что `dry-run` оценивает память;
  - `HF_HOME`/кэш модели, `TEACHER_*` не нужны — предупредить, если сеть недоступна;
  - версия Python, `PYTHONPATH=.pylibs`, наличие `transformers`/`bitsandbytes`.
- Скрипт возвращает код 0/1 и печатает точную следующую команду.
- `scripts/rental_run.sh`: один прогон по шагам (dry-run → 100 шагов → оценка → экспорт),
  с `set -euo pipefail`, логом в `runs/rental/` и остановкой при OOM.
- В `docs/rental-runbook.md`: раздел «первый запуск за 10 минут» — что скопировать,
  что запустить, где смотреть результат, как не потратить лишний час.

## Критерии приёмки

- [ ] `python3 tools/rental_preflight.py` на машине без GPU объясняет, чего не хватает
- [ ] на машине с GPU печатает «можно» и точную команду
- [ ] `scripts/rental_run.sh` проходит `bash -n` и `--dry-run`
- [ ] раздел в рунбуке: команды, ожидаемое время, цена

## Отчёт исполнителя

_Заполняется по завершении._
