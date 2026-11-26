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

Сделано (зона: `eva/tts.py`, `tools/chat_server.py` — только TTS-часть,
`tests/test_tts*.py`, `tests/test_chat_server.py`, `docs/hybrid.md`,
`docs/voice-app.md`; `apps/eva/*`, `src/vtm/*` не тронуты, коммитов нет):

1. **`eva/tts.py` — `ElevenLabsTTS`** (по образцу `PiperTTS`):
   `ELEVENLABS_API_KEY` (обязателен), `ELEVENLABS_VOICE_ID` (по умолчанию
   `EXAVITQu4vr4xnSDxMaL`), `ELEVENLABS_MODEL=eleven_multilingual_v2`,
   `ELEVENLABS_FORMAT=mp3_44100_128`, `ELEVENLABS_TIMEOUT=30`. Запрос:
   `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=…`,
   заголовок `xi-api-key`, JSON `{text, model_id, voice_settings:{stability:0.4,
   similarity_boost:0.75, style:0.0}}`. `synthesize()`/`synthesize_audio()`
   возвращают `(bytes, content_type)` (`audio/mpeg` для mp3); с `out_path` —
   путь. Ключ не входит в `describe()`, `repr(config)` и тексты ошибок.
2. **Приоритет**: `ENGINE_CHOICES` расширен до
   `auto|elevenlabs|piper|system|null`; автовыбор — ElevenLabs (если есть
   ключ) → Piper → say/espeak → null; `EVA_TTS`/`prefer=` перекрывает.
3. **`/api/tts`**: адаптер может вернуть байты или `(bytes, content_type)`;
   MIME определяется **по сигнатуре**: `RIFF…WAVE` → `audio/wav`,
   `ID3`/`0xFFFB` → `audio/mpeg`. Жёсткий `TTS_CONTENT_TYPE` больше не
   используется для ответа (константа оставлена для совместимости).
   `Cache-Control: no-store` сохранён. `default_tts_speech()` теперь отдаёт
   пару `(bytes, content_type)`; пресеты `young/soft` применяются только к
   WAV (MP3 не ресемплируется).
4. **Откат**: при отсутствии ключа, 401, 403, 429, таймауте, сети, обрыве
   чтения или пустом ответе `eva.tts` пишет `logger.warning` с причиной
   (ключ редактируется через `_redact`) и автоматически синтезирует ту же
   фразу Piper'ом (если Piper нет — system). 503 `voice_unavailable`
   возвращается только когда и запасного движка нет; в этом случае текст
   содержит причину, но не ключ.
5. **Кэш**: общий на процесс LRU на 50 успешных ответов, ключ — текст +
   `voice_id` + `model_id` + `output_format` (формат добавлен, чтобы MP3/WAV
   не смешивались). `ELEVENLABS_CACHE=0/false/off` отключает. Есть
   `clear_elevenlabs_cache()` и `elevenlabs_cache_info()` для тестов/диагностики.
6. **Тесты** (`tests/test_tts_elevenlabs.py`, 36 тестов; плюс 6 новых в
   `tests/test_chat_server.py`): URL/заголовки/тело запроса, `voice_settings`,
   content-type, ID3 и `0xFFFB`, кэш (второй вызов не идёт в сеть, LRU 50,
   `ELEVENLABS_CACHE=0`), 401/403/429/таймаут/сеть/пусто, ключ не в тексте
   ошибок и не в логах, откат на Piper при 401/429/таймауте/без ключа, откат
   `synthesize_audio` возвращает `audio/wav`, отсутствие запасного движка —
   ошибка, а не тишина; порядок выбора и перекрытие `EVA_TTS`; серверные
   тесты MP3-ответа, tuple, пустого/не-аудио ответа. Сеть не используется:
   HTTP подменён фейковым opener'ом, `subprocess.Popen` — заглушкой.
7. **Docs**: раздел «ElevenLabs: живой голос с откатом на Piper (задача 0191)»
   в `docs/hybrid.md` — получение ключа, таблица переменных, контракт запроса,
   таблица откатов, кэш/квота, честно про стоимость и приватность (текст
   уходит в облако, сервис платный, кэш только в RAM). В `docs/voice-app.md`
   добавлено, что AVAudioPlayer играет WAV и MP3.

Проверено:

- `python3 -m unittest tests.test_tts tests.test_tts_elevenlabs
  tests.test_chat_server` — **192 теста, OK** (10,3 с), в т.ч. 36 новых
  ElevenLabs-тестов и 6 новых серверных.
- `python3 -m unittest tests.test_tts` — 33 теста, OK (старый контракт
  Piper/system/null не сломан).
- `python3 -m py_compile eva/tts.py tools/chat_server.py` — без ошибок.
- Моками подтверждено: запрос уходит на
  `/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL?output_format=mp3_44100_128`
  с `xi-api-key`, JSON-телом и `voice_settings`; ответ MP3 отдаётся как
  `audio/mpeg`; 401/429/таймаут/сеть/пусто дают лог и WAV от Piper; второй
  одинаковый вызов не идёт в сеть; ключ отсутствует в логах и ошибках.
- Реальный ключ, сеть ElevenLabs и воспроизведение MP3 в браузере/EVA.app
  **не проверялись** (в задаче запрещена реальная сеть; ключа нет).

Риски / что осталось:

- **Нужен ключ от пользователя**: `export ELEVENLABS_API_KEY=...` и, при
  желании, `ELEVENLABS_VOICE_ID` с русским голосом из библиотеки. Дефолтный
  `EXAVITQu4vr4xnSDxMaL` — публичный голос, русский через
  `eleven_multilingual_v2` возможен с акцентом; тембр нужно проверить ушами.
- Реальное качество/задержка ElevenLabs и воспроизведение MP3 в EVA.app не
  проверены — это следующий шаг после получения ключа.
- Пункт 3 исходной формулировки («квота/401/сеть → 503 + откат») выполнен
  по пункту 4: клиент получает **аудио Piper**, а не ошибку; 503 остаётся
  только если Piper/system тоже недоступны. Так пользователь не остаётся без
  голоса.
- `voice_loop.py` при автовыборе с ключом получит MP3-файл (afplay играет,
  но `concat_wavs` для `--out` ожидает WAV). В зону задачи не входило;
  для голосового цикла стоит явно `--tts piper` или доработать склейку.
- 429/401 не кэшируются негативно: каждый запрос снова пробует облако
  (с таймаутом 30 с) и только потом откатывается. Если нужен мгновенный
  откат после серии ошибок — добавить circuit breaker отдельной задачей.
- Кэш хранит байты в памяти процесса (до 50 фраз); при очень длинных фразах
  это единицы мегабайт, но лимит по размеру не задан.
