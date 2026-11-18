# QAT на арендованной GPU: рунбук «взял → обучил → забрал»

Контекст: `docs/distillation.md` (зачем и бюджет), `docs/trit-format.md` (формат тритов); интерфейс — код в `qat/`, источник правды `python -m qat.train --help`.
**«⚠️ сверить»** — не подтверждено репозиторием, рядом команда, показывающая факт.

## 1. Что нужно

| Ресурс | Сколько | Комментарий |
|---|---|---|
| GPU | 24 ГБ (4090/3090) | 1.5B: ~13–15 ГБ с 8-bit Adam, ~22 без (`docs/distillation.md`); 3090 дешевле, но медленнее |
| Диск | ~30 ГБ | веса ~3 ГБ, 2–3 чекпоинта по несколько ГБ, export ~1 ГБ, данные и логи |
| Время | 6–8 ч | обучение ~5–7 ч, остальное — установка, скачивание весов, export |

## 2. Подготовка узла

```bash
sudo apt-get update && sudo apt-get install -y tmux git jq rsync curl
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
pip install -r qat/requirements-qat.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h .; du -sh ~/.cache/huggingface 2>/dev/null || true
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported(), torch.cuda.get_device_capability(0))"
```

Ожидаемо: `True`, `bf16 True`, `cc (8, 9)`, ≥30 ГБ свободно на разделе с `runs/`.
⚠️ сверить колесо torch под драйвер (`pip show torch` + `CUDA Version` в `nvidia-smi`) и `requirements-qat.txt`: он пишется параллельно, ставить по факту импортов, а не по памяти.

## 3. Данные

```bash
make test && make distill-mock                 # конвейер без GPU и без ключей
export TEACHER_BASE=... TEACHER_KEY=... TEACHER_MODEL=...
make distill && wc -l distill/out/pairs.jsonl
head -1 distill/out/pairs.jsonl | jq -e '(.question|strings|length>0) and (.answer|strings|length>0)'
# ~10 тыс. пар на пилот; дальше — на узел: rsync докачивает после обрыва, scp — нет:
rsync -avP --exclude .git --exclude runs --exclude distill/out ./ user@HOST:~/eva32/
rsync -avP distill/out/pairs.jsonl user@HOST:~/eva32/distill/out/
sha256sum distill/out/pairs.jsonl              # тот же на узле — хеши обязаны совпасть
```

`TEACHER_*` читает `distill/teacher.py`, ключ не печатать; в `distill/questions.txt` 9 строк — нужно ~10 тыс., меньше 2–3 тыс. даст переобучение за первую эпоху.
`jq` вернул `true` — поля `question`/`answer` на месте; `false` или пусто — схема другая, править `distill/run.py`.

⚠️ сверить `HOST`, пользователя, порт и путь — из панели провайдера; ssh-ключ уже добавлен.

## 4. Запуск

Сначала `--dry-run` на 3 шага: импорты, конфиг и CUDA ловятся за минуты, а не за часы.

```bash
mkdir -p runs && python -m qat.train --config qat/configs/qwen1.5b.yaml \
  --data distill/out/pairs.jsonl --out runs/dry --max-steps 3 --dry-run 2>&1 | tee runs/dry.log
```

Ожидаемо: конфиг (модель, batch, длина, dtype) и 3 шага без исключений; импорты — §2, память — §6.
Полный прогон — только внутри tmux:

```bash
tmux new -s qat
mkdir -p runs/pilot1
python -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/pairs.jsonl \
  --out runs/pilot1 2>&1 | tee runs/pilot1/train.log
# отцепиться Ctrl-b d, вернуться tmux attach -t qat; прогресс:
tail -f runs/pilot1/log.jsonl | jq -c .
```

⚠️ сверить имена полей `log.jsonl` (`head -1 ... | jq 'keys'`). Время: `шаги = пары × эпохи / (batch × grad_accum)`, затем `шаги × сек/шаг` по логу после 20–30 шагов;
прогноз >10 ч — стоп и уменьшение эпох, а не ожидание пяти часов ради плохой новости.

## 5. Обрывы и возобновление

Чекпоинты каждые N шагов (`ls -la runs/pilot1`, `grep -nE 'save|ckpt|checkpoint|every' qat/configs/qwen1.5b.yaml`); продолжение — тот же `--out` и `--resume`, те же config и data:

```bash
tmux new -s qat            # если прошлая сессия умерла
python -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/pairs.jsonl \
  --out runs/pilot1 --resume 2>&1 | tee -a runs/pilot1/train.log
```

⚠️ сверить, что восстанавливает `--resume` (шаг, эпоху, оптимизатор): нумерация в логе должна продолжиться, а loss — не прыгнуть вверх на порядок.

**tmux обязателен.** SSH-обрыв убивает процесс вместе с сессией, и 8 часов исчезают за секунду; `nohup` — запасной вариант без живого просмотра.
От выключения или пересоздания инстанса не спасают ни tmux, ни nohup: только чекпоинты, уже выкачанные к себе.

## 6. Память и OOM

### 6.1 Формула

Оценка считается до запуска, без torch и без скачивания весов:

```bash
python3 -m qat.train --config qat/configs/qwen3b.yaml --dry-run
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --dry-run
```

`estimate_memory(cfg)` в `qat/config.py` раскладывает пик на пять компонентов
(`N` — параметров, `b` — байт на параметр: fp32 = 4, bf16 = 2):

| Компонент | Формула | Комментарий |
|---|---|---|
| мастер | `N × b(master_dtype)` | сами веса; `master_dtype: bf16` экономит половину |
| рабочая копия | `N × b(compute)`, если `compute ≠ master`, иначе 0 | у bf16-мастера отдельной копии нет |
| оптимизатор | `N × 2` для adam8bit, `N × 8` для adamw | AdamW считается с fp32-состояниями (верхняя оценка) |
| градиенты | `N × b(grad_dtype)` | `grad_dtype: bf16` вдвое меньше fp32 |
| активации | `B×L×b × [H×Layers (checkpointing) \| H×Layers×8 (без)] + B×L×b×(3H+2I) + B×L×b×V + 0.75 ГиБ` | `B` — micro-batch, `L` — seq_len; логиты считаются на всю длину, workspace — запас на контекст и фрагментацию |

Числа в таблицах — **ГиБ** (1 ГиБ = 2³⁰ Б = 1.074 ГБ), как считает PyTorch и
как подписан объём 4090 (24 ГиБ). Арифметика проверяется тестом
`tests/test_qat_memory.py` на трёх конфигурациях по байтам, не «примерно».
«Копия» — консервативный запас: autocast держит bf16-копию одного слоя, а не
всей модели, поэтому реальный пик обычно ниже оценки. В терминах постановки
переход `fp32 → bf16` убирает 5.75 ГиБ мастера и 5.75 ГиБ копии (на 3B).

### 6.2 Таблица: 1.5B/3B × мастер × оптимизатор

Параметры строк: `micro_batch=2`, `seq_len=2048`, `grad_dtype=bf16`,
gradient checkpointing включён — то есть `qat/configs/qwen3b.yaml`.
Для 1.5B это не её пилотный батч (там 4), зато строки сравнимы.

| модель | мастер | оптимизатор | мастер | копия | опт.сост | градиенты | активации | всего | запас до 24 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | fp32 | adam8bit | 5.75 | 2.88 | 2.88 | 2.88 | 2.41 | 16.79 | +7.21 |
| 1.5B | fp32 | adamw | 5.75 | 2.88 | 11.50 | 2.88 | 2.41 | 25.41 | −1.41 |
| 1.5B | bf16 | adam8bit | 2.88 | 0 | 2.88 | 2.88 | 2.41 | 11.04 | +12.96 |
| 1.5B | bf16 | adamw | 2.88 | 0 | 11.50 | 2.88 | 2.41 | 19.66 | +4.34 |
| 3B | fp32 | adam8bit | 11.50 | 5.75 | 5.75 | 5.75 | 2.69 | 31.43 | −7.43 |
| 3B | fp32 | adamw | 11.50 | 5.75 | 22.99 | 5.75 | 2.69 | 48.67 | −24.67 |
| 3B | **bf16** | **adam8bit** | 5.75 | 0 | 5.75 | 5.75 | 2.69 | **19.93** | **+4.07** |
| 3B | bf16 | adamw | 5.75 | 0 | 22.99 | 5.75 | 2.69 | 37.17 | −13.18 |

**Ответ: 3B с bf16-мастером и Adam8bit укладывается — 19.93 ГиБ, запас
+4.07 ГиБ.** При `micro_batch=4` та же конфигурация даёт 21.87 ГиБ (+2.13);
без checkpointing при batch 4 — 29.74 ГиБ, то есть не влезает. Пилот 1.5B
на своём конфиге (`batch=4`, fp32-мастер и fp32-градиенты) — 21.32 ГиБ.

Оценка не заменяет `nvidia-smi`: она не знает про фрагментацию за пределами
0.75 ГиБ, временные буферы fused-ядер и чужие процессы на карте. Если до
лимита меньше ~1 ГиБ — уменьшайте `micro_batch_size`, а не надейтесь.
Старый ориентир §1 «1.5B: ~13–15 ГиБ» не учитывал отдельную bf16-копию:
с копией и fp32-градиентами оценка 21.32 ГиБ, без копии (если бы autocast
не держал её) — 15.58 ГиБ; для планирования берите большую.

### 6.3 Риск bf16-мастера

bf16 хранит 7 бит мантиссы: соседние числа различаются на ≈2⁻⁷ ≈ 0.78 %,
а округление сдвигает не больше чем на 2⁻⁸ ≈ 0.39 %. У веса 0.02 половина
шага ≈ 6e-5. Adam с lr 3e-5 даёт шаг порядка 3e-5 — **меньше половины
шага**, и результат сложения округляется обратно к исходному весу. На 3B
это значит: часть весов (особенно мелкие) может не двигаться вообще, loss
стоит на месте, а не «медленно падает». Чем ближе вес к нулю, тем сильнее
эффект.

Что делать:

* смотреть loss первые 100 шагов; плато при живом lr — первый признак;
* поднять lr выше ulp/|w| (например, 1e-4…3e-4) и проверить, что тернарные
  масштабы не разъезжаются; QAT нестабилен, шаг вверх — по 10–20 шагов;
* альтернативы без bf16-мастера на 24 ГиБ нет: fp32-мастер + bf16-копия +
  Adam8bit + градиенты уже 28.75 ГиБ до активаций, поэтому для 3B на 4090
  это единственный режим; если точность важнее размера — брать 1.5B;
* стохастическое округление обновлений не реализовано: это следующий шаг,
  если bf16-мастер упрётся в плато.

### 6.4 Симптомы и что делать

| Симптом | Причина | Что делать |
|---|---|---|
| `CUDA out of memory` на первом шаге | batch × seq не влезают в 24 ГБ | уменьшить micro-batch и длину, включить gradient checkpointing; 8-bit Adam уже в плане |
| OOM на валидации или в середине | валидация держит граф или больший батч | `torch.no_grad()`, меньше eval-batch; ⚠️ сверить, есть ли валидация в `train.py` |
| OOM при сохранении | лишние копии при сборке state_dict | сохранять под `torch.no_grad()`, `map_location="cpu"`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (⚠️ сверить: `pip show torch`) |
| `bitsandbytes` не ставится | нет колеса под эту пару python/CUDA | `pip show torch bitsandbytes`, колесо под свои версии; ⚠️ сверить матрицу пакета |
| `no kernel image is available` | torch собран под другую архитектуру | `torch.cuda.get_device_capability()`, переустановить torch под 8.9 |
| медленно, загрузка GPU низкая | счёт на CPU, fp32 вместо bf16 или узкое место в данных | `torch.cuda.is_available()` и dtype; `num_workers`, кэш токенизации; ⚠️ сверить ключи конфига |
| loss не падает | LR велик, loss на prompt-токенах, плохие данные | LR вниз в 3–10 раз + warmup, маска `-100` на prompt, 100 пар до переобучения; ⚠️ сверить labels в `train.py` |
| loss = NaN | переполнение или битые пары | клип градиента, пропуск шага, проверка пар из §3 |

## 7. Что забрать с узла

```bash
ls -la export/ && jq . export/manifest.json | head -40
wc -l runs/pilot1/log.jsonl; du -sh export/ runs/pilot1
mkdir -p pilot1 && rsync -avP "user@HOST:~/eva32/export/" pilot1/export/
rsync -avP "user@HOST:~/eva32/runs/pilot1/log.jsonl" "user@HOST:~/eva32/runs/pilot1/train.log" pilot1/
python - <<'EOF'
import json, torch
m = json.load(open('pilot1/export/manifest.json'))
d = torch.load('pilot1/export/ternary.pt', map_location='cpu')
trits = [k for k, v in d['trits'].items()]
print('тензоров с тритами:', len(trits))
bad = [k for k in trits if set(torch.unique(d['trits'][k]).tolist()) - {-1, 0, 1}]
print('вне {-1,0,1}:', bad or 'нет')
print('масштабов:', len(d['scales']), '| блок:', m.get('quant', {}).get('block_size'))
EOF
```

export, manifest и логи обязательны: лог доказывает, что прогон был, и по нему считается цена. `ternary.pt` содержит только троичные тензоры и масштабы; эмбеддинги, norm и `lm_head` остаются в `model.pt` и в экспорт не входят — manifest это фиксирует.
0 тернарных — либо в export упакованные триты (проверять через `tools/trit_codec.py::unpack20` и векторы из `docs/trit-format.md`),
либо `qat/export.py` пишет fp32-мастер без квантования: это баг экспорта, а не «модель не обучилась».

⚠️ сверить имена файлов и схему `manifest.json` по `qat/export.py`: масштабы лежат отдельными положительными float-тензорами (один на блок или канал).

## 8. После: eval на своей машине

```bash
export EVA_BASE=http://127.0.0.1:8000
python3 eval/run_eval.py --model eva32 --repeat 3
ollama pull qwen2.5:1.5b
make eval-dev DEV_MODEL=qwen2.5:1.5b REPEAT=3   # базовая линия Q4
```

⚠️ сверить, чем поднимать export: HF-совместимый safetensors поднимет обычный сервер (без тернарной скорости, пока нет распаковки в ядре), упакованные триты — только наш раннер;
поддержку в vLLM/llama.cpp сверять по их документации.

⚠️ сверить тег (`ollama list`); в `dev/models.lock` 1.5B пока нет. Сравнивать `passed/total`, проваленные id и `checks` при одинаковом `bundle`;
приёмка из плана — не хуже Q4 по фактам и держит инварианты (язык, отказ, «не выдумывать цену»).

## 9. Деньги

`стоимость = часы × ставка + диск/трафик`; ставку брать из панели, ориентир 4090 — 90–120 ₽/ч.

```bash
date -u      # до старта
date -u      # после export: разница = оплаченные часы при поминутном биллинге
```

6–8 ч ≈ **600–900 ₽** (сходится с `docs/distillation.md`). Считать самому: `шаги × сек/шаг / 3600` плюс ~0,5 ч на установку и ~0,5 ч на export и выгрузку.
**Выключить инстанс сразу после rsync**, не до него, иначе увезёте пустой export; «оставил на сутки» — это ×3 к смете.
