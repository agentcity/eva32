---
task_id: "0191"
type: feature
title: ElevenLabs как голос Евы — с откатом на Piper
created_at: 2026-11-26T03:20:00Z
author: yuri
assignee: agent:elevenlabs
status: in_progress
scope: {docs: true, code: true, data: false}
---

# Живой голос вместо робота

## Проблема
Piper разборчив, но звучит механически, и голос не нравится. ElevenLabs даёт
естественную интонацию и русские голоса.

## Требования
1. **Провайдер ElevenLabs** в `eva/tts.py` (по образцу Piper/System):
   - `ELEVENLABS_API_KEY` (обязателен), `ELEVENLABS_VOICE_ID` (по умолчанию
     внятный русский голос), `ELEVENLABS_MODEL` (default `eleven_multilingual_v2`),
     `ELEVENLABS_FORMAT` (default `mp3_44100_128`);
   - `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}` с `xi-api-key`,
     JSON `{text, model_id, voice_settings:{stability, similarity_boost, style}}`;
   - вернуть аудио-байты (mp3) и корректный content-type.
2. **Приоритет движков**: ElevenLabs → Piper → say → espeak → null (явный выбор
   через `EVA_TTS=elevenlabs|piper|...` перекрывает автовыбор).
3. **Сервер** `/api/tts`: отдавать правильный `Content-Type` (audio/mpeg для mp3),
   не ломать клиентов (Web Audio декодирует mp3, AVAudioPlayer тоже).
4. **Деградация**: нет ключа/сеть/квота → понятная ошибка 503 с текстом и
   автоматический откат на Piper (не молчать).
5. **Кэш**: одинаковые фразы не гонять дважды (LRU на 50 записей, по хэшу текста
   + голоса + модели) — экономит квоту.
6. Тесты: мок HTTP-сервер в тестах (без сети), проверка заголовков, тела,
   content-type, откатов, кэша; ключ нигде не логируется.

## Критерии приёмки
- [ ] при `ELEVENLABS_API_KEY` `/api/tts` отдаёт mp3, играется в чате и в EVA.app
- [ ] без ключа — Piper (как сейчас), чат не падает
- [ ] квота/401/сеть → 503 с понятным текстом + откат
- [ ] ключ не попадает в логи и ошибки
- [ ] тесты на моках, полный набор зелёный

## Отчёт исполнителя

_Заполняется по завершении._
