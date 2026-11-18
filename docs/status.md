# Состояние проекта EVA32

Снимок на **2026-11-18 05:14 MSK**. Здесь только навигация и проверяемые факты: у каждой
строки — команда или коммит. Нет подтверждения — значит это план, а не «работает».

Пять фактов, которые важнее остального:

- **Гибрид работает** — 60/60 маршрутов, 188,20 ₽ за 1000 запросов (замер на заглушках, не живой прогон).
- **ВТМ работает на синтетике** — C↔Python max|Δ| = 4,99e-10; настоящая модель в неё ещё не собрана.
- **Троичная модель не обучена** — в репозитории нет ни одного `.trit`/`ternary.pt`.
- **Каталог демонстрационный** — 13 позиций, контакты-заглушки.
- **QAT требует аренды GPU** — без torch/CUDA `--dry-run` выходит с кодом 2.

## Работает

| Что | Чем подтверждено | Подробности |
|---|---|---|
| Гибрид: маршруты и цена | `python3 tools/eval_hybrid.py` → `точность 60/60 = 100.0%`, `гибрид 188.20`, `экономия vs big 63.3%` | [hybrid.md](hybrid.md), [ADR 0009](adr/0009-hybrid-router.md) |
| Скиллы впереди модели | тот же замер → `содержание: ok 24 (100.0%)`, 24/60 ответов без модели | [hybrid.md](hybrid.md) |
| HTTP-сервер: OpenAI API, SSE, RAG, телеметрия | `python3 -m unittest discover -s tests -q` → `Ran 542 tests ... OK`; коммиты `d5fb96e`, `f238948`, `9bdb487` | [hybrid-runbook.md](hybrid-runbook.md), [integration-orioncode.md](integration-orioncode.md) |
| Защита сервера: Bearer-токен и лимит 429 | `python3 -m unittest tests.test_server -q` → `Ran 70 tests ... OK`; `EVA_AUTH_TOKEN` и оба режима описаны в runbook | [hybrid-runbook.md](hybrid-runbook.md), [задача 0030](../ai/tasks/0030-eva-auth.md) |
| Каталог и скиллы на демо-данных | `python3 -c "import json;print(len(json.load(open('data/catalog.json'))))"` → `13`; `grep -n "демонстрационного каталога" eva/skills/contacts.py` → строка 3 | [catalog.json](../data/catalog.json), [задача 0015](../ai/tasks/0015-eva-skills.md) |
| ВТМ: C-интерпретатор, `.vtm`, триты, BPE | `make -C src/vtm test` → `все тесты VTM: ok`; коммиты `3f0f849`, `3da9ec7` | [vtm.md](vtm.md), [src/vtm/README.md](../src/vtm/README.md) |
| Сверка C↔Python на синтетике | `python3 tools/vtm_compile.py --synthetic --mode attn --size 128 --seed 1 --out /tmp/synth.vtm` + `make -C src/vtm test`; max\|Δ\| = 4,99e-10, argmax 48 (attn) и 66 (mlp) — замер при приёмке 0002 | [src/vtm/README.md](../src/vtm/README.md), [задача 0002](../ai/tasks/0002-vtm-runtime-c.md) |
| Кодек тритов | `python3 tools/trit_codec.py` → `самопроверка: ok`, 1,60 бит/трит | [trit-format.md](trit-format.md) |
| Фильтры дистилляции | `python3 -m unittest tests.test_filters -q` → `Ran 21 tests ... OK`; коммит `a7f5b1a` | [distillation.md](distillation.md) |

## В разработке

| Что | Состояние на снимке | Подробности |
|---|---|---|
| 0028: полная модель в `.vtm` | `grep -n "vtm_meta_rope_theta" src/vtm/loader.c` → есть; приёмки и отчёта нет | [vtm.md](vtm.md), [задача 0028](../ai/tasks/0028-vtm-full.md) |
| 0029: компилятор полной модели | `grep -n "def compile_model" tools/vtm_compile.py` → есть; отчёт пуст, теста полной модели нет | [задача 0029](../ai/tasks/0029-vtm-compiler.md) |
| 0031: живой замер качества | `python3 tools/eval_live.py --dry-run` → набор 20 вопросов, `20/20`; живого прогона и трат ещё не было | [задача 0031](../ai/tasks/0031-eva-live-eval.md) |
| 0032: голос ASR→eva→TTS | `python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт" --no-play` работает (pricing → TTS, ~980 мс); ASR не проверен, `docs/voice-prototype.md` нет; контракт — [ADR 0010](adr/0010-voice-pipeline.md) | [задача 0032](../ai/tasks/0032-eva-voice.md) |
| 0033: эта сводка | `grep -n "docs/status.md" README.md` → ссылка из блока «С чего начать» | [задача 0033](../ai/tasks/0033-docs-sync.md) |

## План и заблокированное

Ниже — не «почти готово»: каждая строка либо ждёт аренду/данные, либо ещё не начата.

| Что | Почему не работает | Подробности |
|---|---|---|
| **Троичная модель не обучена** | QAT-прогона не было; `find . -name '*.trit' -o -name 'ternary.pt'` → пусто; задача 0008 `blocked` | [quantization.md](quantization.md), [задача 0008](../ai/tasks/0008-qat-pilot-run.md) |
| **QAT требует аренды GPU** | `python3 -m qat.train --dry-run` → `[qat] не хватает зависимостей: torch, transformers, pyyaml`, код 2; нужна 4090 на 6–8 ч (600–900 ₽) | [qat.md](qat.md) |
| Дистилляция в 1–3B | нет 10 тыс. пар: учитель подключается, но банк вопросов упирается в каталог orioncode (0007 `blocked`) | [distillation.md](distillation.md), [задача 0007](../ai/tasks/0007-catalog-questions.md) |
| Эмбеддинги в тритах | задача 0027 `pending`, три прогона QAT ≈ 2 тыс. ₽, без аренды не стартует | [quantization.md](quantization.md) |
| Полный слой Qwen 1.5B в ВТМ | задача 0006 `pending`: синтетика сошлась, на реальных весах сверки нет | [задача 0006](../ai/tasks/0006-vtm-parity.md) |
| int4/Q4 и калибровка | шаг 1 плана квантования; прогона и чисел нет | [quantization.md](quantization.md) |
| SIMD и троичное железо | после v0.1: сначала NEON/AVX, железо — исследование без срока | [ADR 0003](adr/0003-own-vtm.md) |

## Чего нет (не путать с «работает»)

- обученной троичной модели и её качества на нашем наборе;
- живого замера «гибрид против большой модели» на одних вопросах;
- защиты по умолчанию: без `EVA_AUTH_TOKEN` сервер открыт — токен надо задать самому;
- ASR-замера и документа с числами задержки голосового прототипа;
- реального каталога orioncode: данные демонстрационные, показывать клиенту нельзя;
- SIMD-ядер и любой скорости на настоящей модели в ВТМ.

## Как перепроверить

```bash
python3 tools/eval_hybrid.py                 # 60/60, 188.20 ₽/1000
python3 -m unittest discover -s tests -q     # тесты (542 на снимке)
make -C src/vtm test                         # C-рантайм ВТМ
python3 tools/trit_codec.py                  # кодек тритов
python3 -m unittest tests.test_filters -q    # фильтры дистилляции
python3 -m qat.train --dry-run               # код 2: нужна GPU-машина
```

## Расхождения и риски

- **`eva.cli` не существует.** [hybrid.md](hybrid.md) и [hybrid-runbook.md](hybrid-runbook.md)
  обещают `python3 -m eva.cli ask`; фактический вход — `python3 -m eva` (сервер). Документы расходятся с кодом.
- **60/60 — это маршруты, а не качество ответов.** Текст моделей в замере — заглушка;
  живой замер (0031) не сделан, поэтому «гибрид отвечает лучше» пока не доказано.
- **188,20 ₽/1000 — оценка**, а не счёт провайдера: тариф 0,40/1,20 ₽ за 1k токенов и
  системный промпт ≈ 611 токенов; реальный счёт зависит от токенизатора и длины ответов.
- **Каталог демонстрационный:** контакты и часть данных — заглушки; на реальной выгрузке
  (0007) цифры и ссылки изменятся, а вместе с ними — ответы скиллов и цена.
- **Отчёты отстают от кода:** 0030 уже защищает сервер (70 тестов), но отчёт исполнителя
  пуст; 0028, 0029, 0031, 0032 тоже без отчётов — приёмка ещё не пройдена.
- **Снимок быстро стареет:** параллельные агенты ещё пишут 0028–0032;
  перед релизом перепроверьте командами выше, а не по этому тексту.

## Куда дальше

- [roadmap.md](roadmap.md) — фазы и критерии продукта.
- [work-plan.md](work-plan.md) — очерёдность работ и гейты.
- [adr/README.md](adr/README.md) — все архитектурные решения.
- [../ai/tasks/index.json](../ai/tasks/index.json) — реестр задач и статусы.
