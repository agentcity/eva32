---
task_id: "0193"
type: feature
title: Silero TTS — локальный русский нейроголос вместо Piper (по умолчанию)
created_at: 2026-11-26T03:25:00Z
author: yuri
assignee: agent:silero
status: in_progress
scope: {docs: true, code: true, data: false}
---

# Живой голос без облака

## Зачем
Piper разборчив, но звучит механически. ElevenLabs даёт качество, но платный и
уводит текст в облако (пользователь отказался). Silero — русская нейросеть,
локально, бесплатно, MIT/CC.

## Установка (в vendor/, в git не попадает)
1. `python3.13 -m pip install --target vendor/torch-py torch` (вендорный torch,
   колёса для cp313 macOS arm64). Проверить `import torch` **только из этого каталога**
   (в `.pylibs` лежит сломанный source-checkout — не использовать!).
2. Скачать модель Silero: `torch.hub.load('snakers4/silero-models','silero_tts',
   language='ru', speaker='v4_ru', trust_repo=True)` (первый раз нужна сеть),
   кэш положить в `vendor/silero/` (`TORCH_HOME`), чтобы дальнейшие запуски офлайн.
3. Скрипт установки `tools/install_silero.sh` (идемпотентный, как `install_piper.sh`):
   проверяет python3.13, ставит torch, качает модель, делает пробный синтез
   «Проверка голоса», печатает размеры и время.

## Код
1. `eva/tts.py`: адаптер `SileroTTS` (по образцу PiperTTS):
   - поиск: `SILERO_PYTHON` (по умолчанию python3.13 с PYTHONPATH=vendor/torch-py),
     `SILERO_MODEL_DIR` (vendor/silero), динамический импорт torch только внутри;
   - синтез: 48 кГц, `speaker` из `SILERO_SPEAKER` (по умолчанию `baya` или `xenia`
     — выбрать по замеру/прослушиванию), `sample_rate=48000`, WAV байты;
   - **питч/темп-пресеты не применять** (они были для Piper).
2. Порядок выбора: `EVA_TTS` → ElevenLabs (если ключ) → **Silero (если установлен)**
   → Piper → say → null. Аргумент: локальное лучше облака по умолчанию.
3. Нет torch/модели → откат на Piper с логом (не падать).
4. Первый синтез в процессе прогревает модель ~1–2 с — замерить и закэшировать
   в `describe()`/логах; повторные фразы должны быть быстрыми (<0,5 с на короткую).

## Проверки
- [ ] `synthesize()` даёт WAV 48 кГц, играется в чате и EVA.app (клиент уже умеет)
- [ ] время: короткая фраза <1 с (после прогрева), память процесса +~300 МБ
- [ ] без установки — Piper (как сейчас), тесты зелёные
- [ ] тест с моком torch (без реальной модели) на откат и выбор движка
- [ ] `sh tools/install_silero.sh` воспроизводим, размеры/время печатает

## Отчёт исполнителя

_Заполняется по завершении._
