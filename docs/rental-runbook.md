# Рунбук аренды GPU: 4090, QAT 1.5B/3B

Операционная часть к пилоту [0008](../ai/tasks/0008-qat-pilot-run.md) и эксперименту
[0027](../ai/tasks/0027-trit-embeddings.md): один прогон QAT на арендованной **RTX 4090
24 ГиБ**, от данных до замера качества. Устройство QAT, память и OOM — [docs/qat.md](qat.md);
здесь — карта, установка, данные, порядок шагов, деньги и чек-лист «до оплаты».
Первый запуск одной командой (preflight + дым 100 шагов + val loss) — §0.1.

Снимок на 2026-11-18; раздел §0.1 и команды preflight — 2026-11-23 (задача 0126).
У каждого числа — задача, где оно получено, или команда, которой оно проверяется.
**«⚠️ сверить»** — репозиторием не подтверждено; рядом команда, показывающая факт.

## 0. Что изменилось с прошлой версии

Прошлый рунбук писался до спринтов 6–10 (задачи 0045, 0047, 0077, 0081). Устаревшие
числа заменены — вот они построчно, чтобы не искать глазами:

| Было | Сейчас | Источник |
|---|---|---|
| «3B не влезает в 24 ГБ, нужно 48 ГБ / 2×4090» | **19,93 ГиБ**, запас +4,07 — влезает (bf16-мастер, Adam8bit, checkpointing) | [0047](../ai/tasks/0047-qat-3b-24gb.md) |
| «1.5B: 12,3 ГБ с 8-bit Adam» | **21,32 ГиБ** в пилотном конфиге (fp32-мастер + bf16-копия + fp32-градиенты, batch 4), запас +2,68 | [0047](../ai/tasks/0047-qat-3b-24gb.md) |
| «без bitsandbytes AdamW и 21,6 ГБ» | AdamW на 1.5B — **25,41 ГиБ** при batch 2 и **29,95** на пилотном batch 4 (пересчёт `estimate_memory`), то есть OOM; на 3B — 48,67 | [0047](../ai/tasks/0047-qat-3b-24gb.md) |
| «пар ~10 тыс., меньше 2–3 тыс. — переобучение» | на диске **4008** пар по программированию | [0045](../ai/tasks/0045-code-data.md) |
| «2000 шагов ≈ 6 ч» | 3 эпохи на 4008 парах — **376 шагов**, ≈1,13 ч по оценке 10,8 с/шаг | [0036](../ai/tasks/0036-emb-experiment.md) |
| «сутки — 2400 ₽» | пилот ≈2,1 ч ≈ **192–256 ₽** при 90–120 ₽/ч; сутки не нужны | расчёт §10 |
| данные — `distill/out/pairs.jsonl` | реальные пары — `distill/out/code_pairs.jsonl`; `pairs.jsonl` — 206 мок-пар | [0045](../ai/tasks/0045-code-data.md) |

## 0.1. Первый запуск за 10 минут

Короткий путь: проверить окружение, прогнать 100 шагов и увидеть val loss, а не
испечь час аренды на настройке. Полный пилот — следующим шагом, после того как
дым прошёл. Команды — `tools/rental_preflight.py` и `scripts/rental_run.sh`; пакеты
они не ставят. Preflight не скачивает ничего, а `--dry-run` у `rental_run.sh` при
пустом кеше HF скачает веса (~3 ГиБ, 5–10 мин).

### 0.1.1. Что скопировать

`data/train.jsonl` (3607 пар) и `data/val.jsonl` (401) — split задачи
[0092](../ai/tasks/0092-qat-data-prep.md), ровно под него в конфиге `max_steps: 350`
(3,1 эпохи). `runs/mixture/` (67,5 тыс. пар) — микс для экспериментов: на нём те же
350 шагов дают 0,17 эпохи, preflight об этом предупреждает.

```bash
rsync -avP --exclude .git --exclude .pylibs --exclude .hfcache \
  --exclude 'runs/bench_shape' --exclude 'runs/sweep' --exclude 'runs/tiny*' \
  --exclude 'runs/rental' \
  ./ user@HOST:~/eva32/
ssh user@HOST 'cd ~/eva32 && wc -l data/train.jsonl data/val.jsonl'   # 3607 и 401
```

Зависимости — §2 (`python3 -m pip install -r qat/requirements-qat.txt`), один раз.

### 0.1.2. Что запустить

```bash
cd ~/eva32
python3 tools/rental_preflight.py     # «можно» / «нельзя» + точная следующая команда
bash scripts/rental_run.sh --dry-run  # план без обучения; можно и до оплаты
bash scripts/rental_run.sh            # dry-run → 100 шагов → val loss → стоп
```

Preflight проверяет Python 3.10–3.12, torch/CUDA и имя GPU, свободную и полную VRAM
против оценки **21,32 ГиБ** (запас +2,68), данные и val, конфиг и оценку
`estimate_memory`, `transformers`/`bitsandbytes`/`numpy`/`pyyaml`, кеш HF и свободное
место. Код 0 — «можно», 1 — «нельзя»; в конце печатается одна точная команда запуска.

`rental_run.sh` делает шаги и останавливается: preflight → `qat.train --dry-run`
(3 шага на случайных данных, ловят импорты/CUDA) → 100 шагов на `data/train.jsonl` →
val loss через `tools/eval_torch.py` по `data/val.jsonl`. Всё пишется в
`runs/rental/` — `<id>.log` и каталог `<id>/`.

### 0.1.3. Сколько это займёт и стоит

| Шаг | Шагов | Счёт | С арендой | ₽ при 90–120 ₽/ч |
|---|---:|---:|---:|---:|
| `rental_preflight.py` | — | ~15 с | — | 0 |
| `--dry-run` (веса в кеше) | 3 | 2–5 мин | — | — |
| дым `rental_run.sh` | 100 | 18 мин | ≈1,05 ч | 94–126 |
| **полный пилот** | **350** | **63 мин** | **≈2,1 ч** | **192–256** |

Счёт — по 10,8 с/шаг (замер [0036](../ai/tasks/0036-emb-experiment.md)); после первых
20 шагов возьмите `step_sec` из `log.jsonl` и перезапустите оценку с `--sec-per-step`.
Первый запуск с пустым кешем HF добавляет 5–10 мин на веса (~3 ГиБ) — preflight это
показывает. 350 шагов ≈ 3,1 эпохи на `data/train.jsonl` — это пилот; бюджет §10
считался на 350–376 шагов.

### 0.1.4. Где смотреть результат

```bash
ls -la runs/rental/                                   # <id>.log и <id>/
tail -3 runs/rental/<id>/step100/log.jsonl            # loss, step_sec, gpu_peak_gb
grep -E 'val loss' runs/rental/<id>.log               # итог шага 4
cat runs/rental/<id>/val.json                         # accuracy, perplexity, ppl_pairs
```

`val loss = ln(perplexity)` печатается последним. Сравнивать его надо с base-моделью
на тех же парах (`python3 tools/eval_torch.py --model Qwen/Qwen2.5-1.5B-Instruct
--ppl-data data/val.jsonl`), а не с train loss. Чекпоинт —
`runs/rental/<id>/step100/checkpoints/step_000100`, экспорт —
`python3 -m qat.export --run runs/rental/<id>/step100 --out export/rental` (§6).

### 0.1.5. Как не потратить лишний час

- `rental_preflight.py` — до `tmux` и до старта: он говорит «нельзя» и что именно.
- Прогон только в `tmux` (§5): SSH-обрыв убивает процесс, лог остаётся, шаги — нет.
- `--max-steps` не больше 350: `2000` — это ~18 эпох на 3607 парах (§10.4).
- Сначала дым 100 шагов и val loss; полный пилот — если loss и `step_sec` вменяемые.
- OOM — §12 (`micro_batch_size: 2` в копии конфига), экспорт — §6, выгрузка — §7.
- Выключить инстанс сразу после `rsync`; «оставлю на сутки» — 2160–2880 ₽ (§10).

## 1. Какую карту брать

QAT упирается в память, а не в скорость счёта. Всё, что нужно, влезает в одну 4090 24 ГиБ;
48-гигабайтные карты больше не обязательны — это вывод [0047](../ai/tasks/0047-qat-3b-24gb.md).

| Конфигурация | Всего | Запас до 24 ГиБ | Что брать |
|---|---:|---:|---|
| 1.5B, fp32-мастер, Adam8bit, batch 4 (`qwen1.5b.yaml`) | 21,32 ГиБ | +2,68 | **RTX 4090 24 ГиБ** |
| 3B, bf16-мастер, Adam8bit, batch 2 (`qwen3b.yaml`) | 19,93 ГиБ | +4,07 | **RTX 4090 24 ГиБ** |
| 3B, fp32-мастер, Adam8bit | 31,43 ГиБ | −7,43 | не влезает |
| 3B, fp32-мастер, AdamW | 48,67 ГиБ | −24,67 | не влезает; L40S/A6000 или не запускать |

Числа — из `estimate_memory()` в [qat/config.py](../qat/config.py), проверены тестом
`tests/test_qat_memory.py` по байтам ([0047](../ai/tasks/0047-qat-3b-24gb.md)). Строки 3B —
при `micro_batch=2`, seq 2048, `grad_dtype: bf16`; строка 1.5B — пилотный `batch=4` с
fp32-градиентами, поэтому 21,32 и 19,93 сравнивать напрямую нельзя ([qat.md](qat.md) §6.2).

**Риск 3B на 24 ГиБ честно записан:** bf16 хранит 7 бит мантиссы, шаг Adam при lr 3e-5
меньше половины шага округления — часть весов может замерзнуть, loss встанет на плато.
Смотреть loss первые 100 шагов; если стоит — 1.5B или больший lr
([0047](../ai/tasks/0047-qat-3b-24gb.md), [qat.md](qat.md) §6.3). 3090 — та же память,
но медленнее; ставку и наличие сверять в панели провайдера.

## 2. Установка на узле и первый факт о железе

Python 3.10–3.12 и CUDA-сборка torch (`qat/requirements-qat.txt`); `pip` ниже — на
арендуемой машине, в репозитории ничего не ставить.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h . | tail -1                     # ≥ 40 ГБ на один прогон, ≥ 100 ГБ на три (0027)
sudo apt-get update && sudo apt-get install -y tmux git jq rsync curl build-essential
python3 -m venv .venv && . .venv/bin/activate && python3 -m pip install -U pip
python3 -m pip install -r qat/requirements-qat.txt && python3 -m pip show torch | grep Version
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported(), torch.cuda.get_device_capability(0))"
python3 -c "import transformers, bitsandbytes, yaml; print('ok')"; python3 -m qat.train --help | head -3
```

Ожидаемо: `True`, `bf16 True`, `cc (8, 9)` для 4090, `ok`, справка `qat.train` (модуль,
не `qat/train.py` — прямой запуск файла падает на относительном импорте). Если приехала
CPU-сборка — переустановить torch с `--index-url https://download.pytorch.org/whl/cu124`
(⚠️ сверить тег под драйвер узла); без `bitsandbytes` будет AdamW — 29,95 ГиБ на
пилотном batch 4, то есть OOM ([0047](../ai/tasks/0047-qat-3b-24gb.md)).

## 3. Данные на узле

Данные — 4008 пар «вопрос → ответ» по программированию на русском, 11 815 332 Б,
поля `question`/`context`/`answer`/`meta` ([0045](../ai/tasks/0045-code-data.md),
[status.md](status.md) §Данные). Это меньше «~10 тыс.», на которые рассчитывалась
[0008](../ai/tasks/0008-qat-pilot-run.md), поэтому число эпох считается от 4008, а не от
круглого «3 эпохи по 10 тыс.».

```bash
git clone git@github.com:agentcity/eva32.git eva32 && cd eva32
git rev-parse --short HEAD              # этот же commit — в отчёт
# локально: rsync докачивает после обрыва, scp — нет
rsync -avP --exclude .git --exclude runs --exclude distill/out ./ user@HOST:~/eva32/
rsync -avP distill/out/code_pairs.jsonl user@HOST:~/eva32/distill/out/
sha256sum distill/out/code_pairs.jsonl; wc -l distill/out/code_pairs.jsonl   # 4008
```

⚠️ сверить `HOST`, пользователя и порт — из панели провайдера; хеш и число строк на узле
должны совпасть с локальными. Файл `distill/out/pairs.jsonl` в репозитории — мок на
206 пар: если прогон идёт на нём, три эпохи дадут переобучение за первые сто шагов.

Разбиение train/val и предтокенизация — задача [0092](../ai/tasks/0092-qat-data-prep.md),
она **в работе**: `tools/prepare_qat_data.py` в репозитории пока нет, ссылаться на него
нельзя. Пока его нет, качество меряется по `eval/golden.jsonl` (36 вопросов), а не по
отложенной выборке (§9); после 0092 появится `data/train.jsonl`/`data/val.jsonl` и
`data/stats.json` с распределением длин.

## 4. Dry-run памяти (до оплаты, без torch)

```bash
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --dry-run
python3 -m qat.train --config qat/configs/qwen3b.yaml --dry-run
```

Ожидаемо в конце вывода: `итого 21.32 … запас +2.68` для 1.5B и `итого 19.93 … запас
+4.07` для 3B; матрица 1.5B/3B × fp32/bf16-мастер × adam8bit/adamw печатается первой
([0047](../ai/tasks/0047-qat-3b-24gb.md)). Без torch в конце честное «GPU-стека нет:
показана только оценка памяти» — это норма для ноутбука, на узле с torch dry-run делает
ещё 3 шага на случайных данных и ловит импорты/CUDA за минуты, а не за часы.

Если напечатанное `итого` не совпало с таблицей — правится не карта, а конфиг: это другой
`micro_batch`, `seq_len` или `master_dtype`.

## 5. Обучение

Шаги считаются от данных: `шаги = ceil(пары × эпохи / (micro_batch × grad_accum))`. Для
4008 пар и эффективного батча 32 (4×8 у 1.5B, 2×16 у 3B) это **≈125 шагов на эпоху**
(4008 / 32 = 125,25); 3 эпохи = **376 шагов** (`ceil(4008 × 3 / 32)`). Раннер
[0036](../ai/tasks/0036-emb-experiment.md) в `--dry-run` печатает ровно эти 376 шагов и
сам расчёт `ceil(4008 × 3 / 32)` — это проверка арифметики без GPU.

```bash
tmux new -s qat
mkdir -p runs && mkdir -p runs/pilot1
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/code_pairs.jsonl \
  --out runs/pilot1 --max-steps 376 2>&1 | tee runs/pilot1/train.log
```

`--max-steps 376` обязателен: в конфиге `max_steps: 2000` — это ≈16 эпох по 4008 парам,
то есть переобучение и лишние ~5 ч счёта. Сначала — дым на 20 шагах, чтобы увидеть
`gpu_peak_gb` и `step_sec` на настоящих данных:

```bash
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/code_pairs.jsonl \
  --out runs/smoke --max-steps 20 2>&1 | tee runs/smoke.log
jq -r '[.step, .loss, .step_sec, .gpu_peak_gb] | @tsv' runs/smoke/log.jsonl | tail -3
```

Ожидаемо: 20 шагов без исключений, `gpu_peak_gb` < 21,3 ГиБ (оценка §1) и `step_sec`,
по которому пересчитывается бюджет §10. Для 3B те же команды с `qwen3b.yaml` и
`--out runs/pilot3b`; `step_sec` у 3B будет больше — цифру даёт только этот дым, оценки
в репозитории нет. Первые 100 шагов 3B смотреть на loss: плато при живом lr — признак
замирания bf16-весов ([0047](../ai/tasks/0047-qat-3b-24gb.md)).

Прогресс — `tail -f runs/pilot1/log.jsonl | jq -c .`; чекпоинт каждые 100 шагов,
хранятся 2 последних (`save_every`/`keep_last_checkpoints` в конфиге). Один чекпоинт
1.5B — ~9–10 ГБ (fp32-веса + оптимизатор, комментарий в
[qwen1.5b.yaml](../qat/configs/qwen1.5b.yaml)), отсюда и диск §2.

## 6. Экспорт `.trit`

Экспорт не обучает и не требует GPU-памяти сверх весов: берёт чекпоинт, пишет триты и
manifest. Флага `--quantize-embeddings` у экспорта **нет** — режим эмбеддингов читается
из `run.json` прогона (это же отмечает `tools/parity_real.py`).

```bash
python3 -m qat.export --run runs/pilot1 --out export/ 2>&1 | tee runs/export.log
ls -la export/; du -sh export/
jq '{step: .training.step, model: .base_model, quant: .quantization, files: (.files | keys)}' \
  export/manifest.json
sha256sum export/ternary.trit export/non_ternary.bin export/manifest.json
```

Ожидаемо: `ternary.trit` (упакованные триты, ~1 ГБ по [qat.md](qat.md) §1),
`non_ternary.bin` (эмбеддинги, финальная норма, lm_head, bias в fp32) и `manifest.json`
с шагом, конфигом и контрольными суммами ([qat/export.py](../qat/export.py),
[0043](../ai/tasks/0043-trit-embeddings-e2e.md)). Хеши — в отчёт: по ним потом
сверяется, что на своей машине считали те же веса.

## 7. Компиляция в `.vtm`

```bash
python3 tools/vtm_compile.py --manifest export/manifest.json --out export/model.vtm
make -C src/vtm
./src/vtm/vtm inspect export/model.vtm
./src/vtm/vtm run export/model.vtm --tokens 9707,11,1879 --max-new 4 --stats
```

`--tokens` — только для проверки, что контейнер грузится и считает; осмысленный текст
требует `--text` и настоящего `tokenizer.json` (формат HuggingFace BPE поддерживает
[0012](../ai/tasks/0012-bpe-tokenizer.md); взять файл из HF-снапшота модели, ⚠️ сверить,
что он читается, — в репозитории лежит только тестовая фикстура).

Ограничения компилятора, записанные в [0058](../ai/tasks/0058-parity-real.md):
fp32-эмбеддинги 1.5B требуют 8–10 ГиБ RAM в Python-компиляторе (fallback — тритовые
эмбеддинги, [0043](../ai/tasks/0043-trit-embeddings-e2e.md)); `RMSN eps` кодируется как
`imm/1000`, поэтому Qwen-овский 1e-6 честно округляется до 0; на 1.5B скалярное ядро
считает промпт минутами. Числа «сколько стоит проход» на настоящих весах — **не
измерены**, [status.md](status.md) §ВТМ.

## 8. `parity_real`: сверка рантайма с HF

```bash
python3 tools/parity_real.py --dry-run
python3 tools/parity_real.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "The capital of France is" --limit-tokens 16 \
  --quantize-embeddings none --workdir data/parity_real
jq '{status, passed, max_abs_delta: .comparison.max_abs_delta,
     argmax_match: .comparison.argmax_match}' data/parity_real/parity.json
```

Порог: argmax совпал, `max|Δ|` логитов < 1e-2 ([0058](../ai/tasks/0058-parity-real.md)).
Если не прошёл — повторить с `--scale-dtype float32`: из замера уйдёт округление
масштабов до fp16 и останется только расхождение рантайма.

**Важно, чтобы не потратить час впустую:** `parity_real.py` собирает QAT-чекпоинт
**сам**, из весов HF (`apply_ternary`), и делает свой экспорт и компиляцию; `runs/pilot1`
он не читает. То есть это проверка C-рантайма на тех же весах, а не замер нашего
обученного прогона. Качество нашего прогона — следующий шаг, `eval_torch`; сверки логитов
для своего чекпоинта в репозитории пока нет.

## 9. `eval_torch`: качество до/после

```bash
python3 tools/eval_torch.py --dry-run
python3 tools/eval_torch.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --golden eval/golden.jsonl --ppl-data distill/out/code_pairs.jsonl --out runs/eval-base.json
python3 tools/eval_torch.py --model runs/pilot1 \
  --golden eval/golden.jsonl --ppl-data distill/out/code_pairs.jsonl --out runs/eval-pilot1.json
```

`--model runs/pilot1` берёт последний чекпоинт и параметры квантования из
`trainer_state.pt` ([0046](../ai/tasks/0046-eval-torch.md)); `--golden` — 36 вопросов с
поведенческими проверками, `--ppl-data` — те же 4008 пар на perplexity (по умолчанию
200 штук). В JSON: `accuracy`, `passed/total`, `perplexity`, `avg_len` — их и сравнивать,
а не loss. `--local-files-only` — когда веса уже в кеше: экономит сеть и время.
Базовая модель считается на той же машине и том же golden, иначе разница «до/после»
смешана с разницей данных.

## 10. Сколько это стоит

`₽ = часы × ставка`; ставка — из панели, ориентир 4090 — 90–120 ₽/ч
([qat.md](qat.md) §9). Часы = `шаги × step_sec / 3600` + 0,5 ч на подготовку узла +
0,5 ч на экспорт (те же 0,5+0,5 в [qat.md](qat.md) §9); `step_sec` — из дыма §5, по
умолчанию в оценке 10,8 с ([0036](../ai/tasks/0036-emb-experiment.md): 2000 шагов ≈ 6 ч).

| Сценарий | Шагов | Счёт, ч | С арендой, ч | ₽ при 90–120 ₽/ч |
|---|---:|---:|---:|---:|
| **пилот**: 1.5B, 3 эпохи (376) | 376 | 1,13 | 2,13 | 192–256 |
| 3B, 3 эпохи (376, тот же батч) | 376 | 1,13·k | 2,13·k | 192–256, k>1 — по замеру §5 |
| «как в конфиге»: 2000 шагов | 2000 | 6,00 | 7,00 | 630–840 |
| три прогона 0027 (none/int8/trit) | 1128 | 3,38 | 5,38 | 484–646 |
| «оставил на сутки» | — | — | 24 | 2 160–2 880 |

`k` у 3B — отношение `step_sec` 3B к 1.5B; в репозитории его нет, поэтому бюджет 3B
считать после дыма, а не по этой строке. Строка 0027 — из `--dry-run` раннера: 4,88 ч и
488 ₽ на три прогона плюс 0,5 ч подготовки; eval в смету не входит, он идёт дома
([qat.md](qat.md) §8). Свои числа: `--sec-per-step` и `--rub-per-hour` у раннера;
тот же план печатает его `--dry-run`.

Эксперимент 0027 после пилота, если нужно сравнение эмбеддингов (три прогона, конфиг
копируется с подменой `quantize_embeddings`):

```bash
tmux new -s emb
python3 tools/embedding_experiment.py --data distill/out/code_pairs.jsonl \
  --out runs/emb --epochs 3 2>&1 | tee -a runs/emb.log
```

Прогоны лягут в `runs/emb/none|int8|trit`, конфиги — в `_configs/`, логи — в `_logs/`;
`--resume` пропускает готовые, `--keep-going` не бросает остальные режимы. Перед стартом
проверить `--dry-run`: он печатает 376 шагов и смету ([0036](../ai/tasks/0036-emb-experiment.md)).

## 11. Чек-лист перед тем, как платить

- [ ] `python3 tools/rental_preflight.py` — «можно»; если «нельзя», исправить и
      повторить (§0.1), а не запускать обучение «на удачу».
- [ ] `nvidia-smi` — 4090 (cc 8.9), ≥ 23 ГиБ свободных, драйвер записан; `df -h .` — ≥ 40 ГБ
      (один прогон) или ≥ 100 ГБ (три прогона 0027).
- [ ] `torch.cuda.is_available()` → `True`, `is_bf16_supported()` → `True`; `bitsandbytes`
      импортируется (иначе AdamW и ≥25,41 ГиБ — OOM).
- [ ] `python3 -m qat.train --config … --dry-run` даёт ожидаемое: **21,32 (+2,68)** для 1.5B
      или **19,93 (+4,07)** для 3B; другое число — не тот конфиг.
- [ ] `sha256sum distill/out/code_pairs.jsonl` на узле = локальному, `wc -l` = 4008; прогон
      не на 206 мок-парах.
- [ ] дым-прогон 20 шагов без OOM: `gpu_peak_gb` известен, `step_sec` записан, бюджет §10
      пересчитан по нему.
- [ ] `--max-steps 376` (3 эпохи), а не 2000 из конфига: 2000 — это ≈16 эпох на 4008 парах.
- [ ] прогон внутри `tmux`; `save_every: 100`, `keep_last_checkpoints: 2`.
- [ ] для 3B: loss первых 100 шагов не на плато (bf16-мастер,
      [0047](../ai/tasks/0047-qat-3b-24gb.md)).
- [ ] ставка и биллинг из панели записаны в отчёт; `date -u` зафиксирован до старта.
- [ ] план выгрузки: `export/` + `runs/*/log.jsonl` + `train.log` по rsync — до выключения
      инстанса, не после.

## 12. Если OOM

| Симптом | Причина | Что делать |
|---|---|---|
| OOM на первом шаге | micro-batch × seq не влезают в 24 ГиБ | `micro_batch_size: 2` (затем 1): по `estimate_memory()` на этом конфиге 1.5B даёт **19,66 ГиБ** (+4,34) и **18,83** (+5,17), 3B — **18,96** (+5,04); на 3B batch 4 — 21,87 (+2,13). Это оценка, а не замер: печатает `--dry-run` на копии конфига |
| OOM в середине прогона | выросла длина пары или копия при сохранении | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (⚠️ сверить по `pip show torch`) |
| 25–30 ГиБ занято, до OOM близко | `adam8bit` не подхватился, работает AdamW | `python3 -c "import bitsandbytes"`, `pip show bitsandbytes` |
| `no kernel image is available` | torch собран под другую архитектуру | `torch.cuda.get_device_capability()`, переставить torch под cc 8.9 |

Батч — не универсальное лекарство: у 1.5B мастер, bf16-копия, состояния оптимизатора и градиенты
занимают 17,25 из 21,32 ГиБ, поэтому batch 4 → 2 экономит 1,66 ГиБ, а не половину. Таблица
[qat.md](qat.md) §6.2 с «16,79 ГиБ при batch 2» — другой режим (`grad_dtype: bf16`),
с пилотным fp32-конфигом не сравнивать.

```bash
cp qat/configs/qwen1.5b.yaml qat/configs/emb-oom.yaml
$EDITOR qat/configs/emb-oom.yaml
python3 -m qat.train --config qat/configs/emb-oom.yaml --data distill/out/code_pairs.jsonl \
  --out runs/oom-check --max-steps 20
```

Крутить только в копии конфига — baseline `qwen1.5b.yaml` остаётся тем, что сравнивается;
`gpu_peak_gb` в логе подтверждает, стало ли легче. Формула и разбор компонентов —
[qat.md](qat.md) §6.

## 13. Если прогон прервался

`qat.train` по SIGINT/SIGTERM завершается после текущего шага и сохраняет чекпоинт,
поэтому сигнал лучше `kill -9`; чекпоинты — `runs/pilot1/checkpoints/step_000100`.
Продолжение — тот же `--out`, `--config`, `--data` и `--resume` без значения.

```bash
tmux new -s qat
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/code_pairs.jsonl \
  --out runs/pilot1 --resume 2>&1 | tee -a runs/pilot1/train.log
jq -r '.latest_step, .status' runs/pilot1/run.json
```

Ожидаемо: нумерация в `log.jsonl` продолжается, `latest_step` в `run.json` растёт, `loss`
не прыгает вверх на порядок. `--allow-data-change` — только если файл данных менялся.
От смерти инстанса не спасают ни tmux, ни nohup — только чекпоинты, выкачанные к себе.

## 14. Что сверить и чего не проверяли

- CUDA-тег колеса torch под драйвер узла — `pip show torch`, `nvidia-smi` (§2).
- `HOST`/пользователь/порт/путь — панель провайдера (§3).
- `tokenizer.json` из HF-снапшота читается BPE-токенизатором 0012 — `vtm run --text`
  на коротком промпте (§7); в репозитории только фикстура.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` на этой сборке torch (§12).
- `--resume` восстанавливает эпоху и оптимизатор — `run.json`, хвост `log.jsonl` (§13).
- **Не измерено:** реальный пик VRAM, сходимость и качество после QAT, `step_sec` для 3B,
  скорость C-рантайма на весах 1.5B ([status.md](status.md) §QAT, §ВТМ). Эти числа
  появляются только на аренде и должны попасть в отчёт, а не в план.
