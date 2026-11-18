---
task_id: "0096"
type: feature
title: eva — включить кэш префикса в Python-бэкенде
created_at: 2026-11-18T14:20:00Z
author: yuri
assignee: ""
status: pending
scope:
  docs: false
  code: true
  data: false
---

# Кэш префикса в eva

## Проблема

C-ABI получил `cache_prefix` (0087, 125× на prefill-доминантном запросе), но
`eva/backends/vtm_backend.py` его не выставляет — Python-путь работает как раньше, без
выигрыша.

## Требования

- В `eva/backends/vtm_backend.py`: заполнять `cache_prefix` из конфига
  (`EVA_VTM_CACHE=1` или параметр), по умолчанию выключено (не менять поведение).
- Раскладка `vtm_api_options` уже совместима (reserved[6] → reserved[5] + cache_prefix),
  но ctypes-структуру надо переписать: сейчас там 6 полей reserved.
- Тест: два запроса с общим префиксом 64+ токенов на demo-модели — второй быстрее,
  ответы совпадают.

## Критерии приёмки

- [ ] ctypes-структура соответствует api.h (sizeof и смещения)
- [ ] второй запрос с общим префиксом быстрее (число в отчёте)
- [ ] без опции поведение прежнее
- [ ] полный Python-набор зелёный

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `ai/tasks/0087-api-cache.md`, `ai/tasks/0089-eva-vtm-backend.md`
