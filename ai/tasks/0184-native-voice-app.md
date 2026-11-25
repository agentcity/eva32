---
task_id: "0184"
type: feature
title: EVA.app — лёгкое нативное приложение с голосом (STT + TTS)
created_at: 2026-11-26T01:10:00Z
author: yuri
assignee: agent:eva-app
status: in_progress
scope: {docs: true, code: true, data: false}
---

# Своя программа вместо браузера

## Зачем
Браузер тяжёлый, а голос нужен: сказать вслух → текст улетел Еве → ответ
озвучивается. Всё локально, без облака.

## Что проверено (до задачи)
- Swift 6.2 и Xcode есть; `swiftc` компилирует, но модульный кэш надо уводить
  (`SWIFT_MODULECACHE_PATH`/`CLANG_MODULE_CACHE_PATH`), иначе `Operation not permitted`.
- Apple Speech (`SFSpeechRecognizer`, ru-RU, `requiresOnDeviceRecognition`) —
  полностью локальное распознавание, без скачивания моделей и без облака.
  **CLI-бинарю авторизацию не дают** — нужен .app-бандл с Info.plist
  (`NSSpeechRecognitionUsageDescription`, `NSMicrophoneUsageDescription`).
- Piper у нас уже стоит (голос ru_RU-irina, пресеты «моложе/мягче»),
  сервер отдаёт WAV по `POST /api/tts`.
- Чат-сервер: `POST /api/chat` (SSE), порт по умолчанию 8765, колонка «нет»
  отключает вторую половину сравнения.

## Объём
1. **EVA.app** — bundle (Info.plist, ad-hoc подпись не нужна) со статус-баром.
2. **Голосовой ввод**: горячая клавиша (например ⌥Space) или клик — запись с
   микрофона через `AVAudioEngine`, живое распознавание ru-RU на устройстве,
   по отпусканию — текст отправляется в `/api/chat`.
3. **Ответ**: поток SSE отображается в маленьком окне; по желанию — озвучка
   ответа (скачать WAV `/api/tts` и проиграть `AVAudioPlayer`).
4. **Настройки**: адрес сервера, включить озвучку, выбор пресета голоса.
5. Никаких внешних зависимостей, приложение ≤20 МБ, запуск без терминала.

## Критерии приёмки
- [ ] `EVA.app` собирается одной командой (`tools/build_eva_app.sh`), запускается
- [ ] нажал ⌥Space, сказал «привет», текст появился в поле и ушёл Еве
- [ ] ответ печатается (SSE) и, если включено, звучит голосом
- [ ] разрешения запрашиваются один раз, в Info.plist есть оба Usage Description
- [ ] README: как собрать и как пользоваться

## Отчёт исполнителя

Сделано (зона: `apps/eva/*`, `tools/build_eva_app.sh`, `docs/voice-app.md`;
`src/vtm/*`, `tools/chat_server.py`, `eva/*` — только чтение API, не тронуты):

1. **`tools/build_eva_app.sh`** — одна команда собирает `build/EVA.app` из
   `apps/eva/Sources/*.swift` одним `swiftc` (Swift 6.2, языковой режим 6,
   `-target arm64-apple-macosx13.0`); модульный кэш уведён в
   `/tmp/eva-app-{swift,clang}-modulecache`; Info.plist копируется и
   проверяется `plutil`; ad-hoc подпись `codesign -s -` + `codesign --verify
   --strict`; печатается размер. Флаги `--run`, `--clean`, переопределение
   `EVA_OUT_DIR`. Итог — **≈0,4 МБ** (412 КБ на диске; лимит задачи 20 МБ).
2. **`apps/eva/Info.plist`** — `LSUIElement` (статус-бар без иконки в доке),
   `NSMicrophoneUsageDescription`, `NSSpeechRecognitionUsageDescription`,
   `CFBundleIdentifier=com.eva32.eva`, минимум macOS 13.
3. **Приложение** (`apps/eva/Sources`, 13 файлов, ~2,5 тыс. строк):
   - статус-бар (`NSStatusBar`; иконка waveform → mic.fill → speaker.wave.2),
     меню: «Слушать (⌥Space)», «Спросить текстом…», «Настройки…», «Выход»;
   - глобальный ⌥Space через Carbon `RegisterEventHotKey` (без «Универсального
     доступа»); если клавиша занята — в журнале «хоткей: ⌥Space занят», меню
     продолжает работать;
   - запись: `AVAudioEngine.installTap` → `SFSpeechAudioBufferRecognitionRequest`
     (ru-RU, `requiresOnDeviceRecognition = true`, частичные результаты),
     повторное ⌥Space отдаёт последний распознанный текст; авто-стоп через
     60 с; понятные ошибки, если нет прав / офлайн-модели ru-RU / микрофона;
   - чат: `POST /api/chat` с телом
     `{"question": …, "sides": [{"backend","model"}, {"backend": "none"}],
     "max_tokens": 256, "use_system": true}`; разбор SSE `data: {json}`
     (`start/delta/done/error/end`, `[DONE]`) через `URLSession.bytes`;
     `<think>…</think>` вырезается на лету (включая теги, разрезанные по
     чанкам), складывается в свёрнутую строку «размышления (N симв.)» и
     никогда не озвучивается;
   - окно (NSPanel): живой частичный текст, поток ответа, галочка
     «Озвучивать», пресет голоса, «Стоп», «Настройки…», поле ввода + Enter;
   - озвучка: ответ режется на фразы (`. ! ? …`, переводы строк, ≤560 символов
     — у сервера лимит 600), каждая уходит в `POST /api/tts` и играется по
     очереди одним `AVAudioPlayer`; сбой TTS не ломает печать текста;
   - настройки в `UserDefaults`: `eva.serverURL`, `eva.modelSpec`,
     `eva.voice`, `eva.speakReplies`, `eva.maxTokens`; список моделей — из
     `GET /api/backends`;
   - журнал `~/Library/Logs/EVA/eva.log` + stderr (работает без терминала).
4. **Самопроверки без UI и микрофона** (те же классы, что в окне):
   `--selftest-chat`, `--selftest-tts`, `--selftest-backends`,
   `--selftest-speech`, `--selftest-text`; плюс `--ask`, `--no-voice`,
   `--server`, `--model`, `--voice`.
5. **`docs/voice-app.md`** — сборка, первый запуск и разрешения, хоткей,
   настройки, контракт запроса, команды проверки, ограничения.

Чем проверено:

- `sh tools/build_eva_app.sh` → `build/EVA.app (0.4 МБ)`; `codesign --verify
  --strict` → «valid on disk / satisfies its Designated Requirement»;
  `plutil -p` → оба Usage Description, `LSUIElement => true`,
  `CFBundleIdentifier => com.eva32.eva`; сборка «с нуля» в
  `EVA_OUT_DIR=/tmp/...` тоже проходит (`--version` из неё работает).
- `open build/EVA.app` → `pgrep -fl` видит процесс, `lsappinfo info -app
  com.eva32.eva` → `type="UIElement"`, журнал:
  «статус-бар: пункт создан (иконка waveform, пунктов меню 5)»,
  «хоткей: ⌥Space зарегистрирован», «запуск … args=[]». Окон при старте нет
  (0 окон в CGWindowList) — приложение живёт в статус-баре, как задумано.
- Окно: запуск с `--ask` → CGWindowList видит окно `Ева` 580×524.
- Поток `/api/chat` (тестовая мини-модель): `sse start` → первый `delta`
  через 1,5 с → 240 кусков → `done 5.9 с · 349.7 ток/с` → полный ответ.
- Реальная модель `runs/q4/3b-q4k.vtm` через приложение: `sse start` → первый
  `delta` через **29,26 с** → `done 29.9 с · 21.0 ток/с · 15 токенов` →
  `ответ: Привет! Как я могу помочь вам сегодня?`. Под нагрузкой (load average
  17) первый токен приходил и за 75,7 с — приложение ждёт и показывает
  «отвечает: …», сервер при этом обработал запрос (`requests` 8 → 13,
  `last.elapsed_ms = 29881` совпадает с журналом).
- `/api/tts`: `--selftest-tts` → 54 142 байта, заголовок RIFF, `file` →
  «WAVE audio, Microsoft PCM, 16 bit, mono 22050 Hz»; из приложения —
  «озвучка: синтез 135 симв., голос young» → «озвучка: играю 377 280 байт,
  старт=true».
- Скрытие `<think>` и нарезка фраз: `--selftest-text` — 6/6 OK, включая теги
  на границах чанков, незакрытый `<think>` и длинный текст (части ≤560).
- `--selftest-speech`: `ru-RU: available=true onDevice=true`,
  `authorization: notDetermined` (запрос прав — при первом ⌥Space).

Не проверено (нужен человек) и что именно нажать:

- **Реальное распознавание речи.** Авторизацию Apple Speech получает только
  .app-бандл, запущенный через LaunchServices. EVA.app уже запущена (`open
  build/EVA.app`, pid 3590) и висит в статус-баре. Порядок:
  1. нажать **⌥Space** один раз — macOS спросит «Микрофон» и «Распознавание
     речи», разрешить оба;
  2. сказать «привет, как тебя зовут»;
  3. нажать **⌥Space** ещё раз (или «Стоп») — текст уйдёт в чат;
  4. ответ напечатается; с включённой галочкой «Озвучивать» — прозвучит.
  Если прав нет: `tccutil reset Microphone com.eva32.eva` и
  `tccutil reset SpeechRecognition com.eva32.eva`, затем запустить снова.
- Слышимость озвучки: WAV получен, `AVAudioPlayer.play()` вернул `true`, но на
  слух в headless-сессии не проверялось.
- Клики в окне настроек (список моделей из `/api/backends`, сохранение)
  автоматизированно не проверялись; значения `UserDefaults` проверены через
  `defaults read` и по журналу после перезапуска.
- Отрисовка свёрнутых «размышлений» на реальном ответе с `<think>` — логика
  покрыта `--selftest-text`, но у 3b-q4k в проверенных ответах `<think>` не
  было.

Критерии приёмки:

- [x] `EVA.app` собирается одной командой, запускается (`tools/build_eva_app.sh`,
      `open`)
- [ ] нажал ⌥Space, сказал «привет», текст ушёл Еве — **проверяет человек**
- [x] ответ печатается по SSE; озвучка получает WAV и запускает плеер
      (на слух — в ручном тесте)
- [ ] разрешения запрашиваются один раз — оба Usage Description в Info.plist
      есть, сам запрос возникает при первом ⌥Space (ручной тест)
- [x] README: `docs/voice-app.md` — сборка, разрешения, хоткей, проверки

Риски:

- **Один C-контекст VTM**: отменённый клиентом запрос продолжает считаться на
  сервере, следующий вопрос ждёт (наблюдали 29–77 с до первого токена при
  load average 17). Приложение старый поток отменяет и не блокирует новый
  вопрос, но серверная очередь — вне зоны задачи; в журнале видно
  «стоп по кнопке».
- **Ad-hoc подпись**: после каждой пересборки cdhash меняется, TCC может
  спросить разрешения заново.
- `open --args` в этой среде не передал аргументы (`args=[]`), поэтому `--ask`
  проверялся запуском бинаря из бандла. На обычный запуск через `open` это не
  влияет: вопрос задаётся голосом или из поля ввода.
- ⌥Space может быть занят другим приложением — тогда в журнале «занят», меню
  остаётся.
- Тестовая `runs/tiny/model.vtm` выдаёт мусор (её нормальное поведение) — на
  ней проверен только протокол; содержательный ответ получен на 3b-q4k.
