---
task_id: "0197"
type: feature
title: Голос — только Silero (baya); Piper/Ирину убрать
created_at: 2026-11-26T04:40:00Z
author: yuri
assignee: agent:silero-only
status: in_progress
scope: {docs: true, code: true, data: false}
---

# Один голос вместо зоопарка

## Решение пользователя
Silero (baya) принят. Старый Piper/Ирина больше не нужен — убрать, чтобы не
плодить настройки и путаницу.

## Требования
1. **Выбор голоса = голоса Silero**: `baya` (по умолчанию), `kseniya`, `xenia`,
   `aidar`. В чате селектор «голос» должен показывать их, а не пресеты
   `young/soft/irina`.
2. **Убрать питч-пресеты Piper** из `tools/chat_server.py` (`VOICE_PRESETS`,
   `_pitch_shift_wav`, `_smooth_wav`, ветки применения) — они были только для WAV
   от Piper и к Silero не относятся.
3. **Порядок движков**: `EVA_TTS` → ElevenLabs (если ключ) → **Silero** → system
   (`say`) → null. Piper из автовыбора убрать (код адаптера может остаться, но
   не выбирается и не документируется).
4. **Убрать установку Piper**: `tools/install_piper.sh`, `vendor/piper*`,
   `samples/voice_piper_*.wav`, `samples/voice_milena.wav`; README/доки — раздел
   про Piper заменить на Silero (или удалить).
5. **EVA.app**: настройки голоса — список голосов Silero (`baya` по умолчанию);
   пресеты young/soft убрать из UI, значения в UserDefaults миграция: если было
   `young/soft/irina` → `baya`.
6. Проверить сквозной путь: `/api/tts {"voice":"kseniya"}` даёт WAV другим голосом;
   приложение говорит выбранным; при отсутствии Silero — `say` (Milena), не падает.

## Критерии приёмки
- [ ] селектор голоса: baya/kseniya/xenia/aidar, по умолчанию baya
- [ ] питч-пресеты и их код удалены, тесты обновлены
- [ ] Piper не устанавливается и не упоминается в инструкциях
- [ ] `vendor/piper*` и сэмплы Piper удалены (репозиторий чище)
- [ ] полный набор тестов зелёный, `sh tools/build_eva_app.sh` без ошибок

## Отчёт исполнителя

_Заполняется по завершении._
