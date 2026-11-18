---
task_id: "0032"
type: feature
title: Голос: прототип ASR → eva → TTS
created_at: 2026-11-18T05:30:00Z
author: yuri
assignee: agent:eva-voice
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Голос: прототип ASR → eva → TTS

Контракт — `docs/hybrid.md`, спеки — `docs/vtm.md`, ADR в `docs/adr/`.

## Проблема

ADR 0010 описывает конвейер, но кода нет. Нужно проверить, что связка работает и
замерить задержку на реальном железе.

## Требования

- `scripts/voice_loop.py`: WAV на входе → whisper.cpp (если собран) → `eva.cli ask` или HTTP → `say -v Milena` → WAV/выход.
- Если whisper.cpp нет — честное сообщение и путь `--text` для отладки цепочки без ASR.
- Замер каждого этапа: ASR, маршрут, TTS; итог до первого звука.
- `docs/voice-prototype.md`: как собрать whisper.cpp, как запустить, что получилось (числа).

## Критерии приёмки

[ ] `python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт"` озвучивает ответ через say
[ ] замер этапов выводится (мс)
[ ] без whisper.cpp — понятное сообщение, не падение
[ ] документ с реальными числами задержки

## Контекст и ограничения

- Зона: scripts/voice_loop.py, docs/voice-prototype.md, tests/test_voice_loop.py
- Другие файлы не трогать: параллельно работают другие агенты.
- Только stdlib; сеть в тестах запрещена (кроме живого замера 0031, и то по флагу).
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`, `docs/vtm.md`
