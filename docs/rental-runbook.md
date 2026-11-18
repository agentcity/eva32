# Рунбук аренды GPU под эксперимент 0027

Операционная часть к [задаче 0027](../ai/tasks/0027-trit-embeddings.md): три прогона QAT (эмбеддинги `none` / `int8` /
`trit`) на одном наборе. Прогон, чекпоинты и экспорт — в [docs/qat.md](qat.md); здесь только аренда: карта, установка,
доставка, порядок, деньги. **«⚠️ сверить»** — не подтверждено репозиторием, рядом команда, показывающая факт.

## 1. Какую карту брать
QAT упирается в память, а не в скорость счёта (`docs/distillation.md`, «Железо для QAT»).

| Модель | Что нужно | Что брать |
|---|---|---|
| 1.5B (эксперимент 0027) | 12,3 ГБ с 8-bit Adam + 1–3 ГБ активаций | **RTX 4090 24 ГБ** (или 3090); 4090 быстрее |
| 3B | 43,3 ГБ без 8-bit Adam, 24,7 ГБ с ним — впритык | **48 ГБ**: L40S или A6000; либо 2×4090 с ZeRO-2 |

На 24 ГБ 3B не запускать: 24,7 ГБ — только веса и оптимизатор, без активаций; H100 памяти не добавляет.

## 2. Установка на узле и первый факт о железе
Python 3.10–3.12 и CUDA-сборка torch (`qat/requirements-qat.txt`); `pip` ниже — на арендуемой машине.
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h . | tail -1                     # ≥ 80 ГБ: ~30 ГБ на прогон, прогонов три
sudo apt-get update && sudo apt-get install -y tmux git jq rsync curl
python3 -m venv .venv && . .venv/bin/activate && python3 -m pip install -U pip
python3 -m pip install -r qat/requirements-qat.txt && python3 -m pip show torch | grep Version
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
python3 -c "import transformers, bitsandbytes, yaml; print('ok')"; python3 -m qat.train --help | head -3
```
Ожидаемо: `True`, `(8, 9)` для 4090, `ok`, справка `qat.train`. Если приехала CPU-сборка — переустановить torch с
`--index-url https://download.pytorch.org/whl/cu124` (⚠️ сверить `cu124`); без `bitsandbytes` будет AdamW и 21,6 ГБ — OOM.

## 3. Репозиторий и данные на узле
```bash
git clone git@github.com:agentcity/eva32.git eva32 && cd eva32
git rev-parse --short HEAD              # этот же commit — в отчёт
# локально: rsync докачивает после обрыва, scp — нет
rsync -avP --exclude .git --exclude runs --exclude distill/out ./ user@HOST:~/eva32/
rsync -avP distill/out/pairs.jsonl user@HOST:~/eva32/distill/out/
sha256sum distill/out/pairs.jsonl; wc -l distill/out/pairs.jsonl
```
⚠️ сверить `HOST`, пользователя и порт — из панели провайдера; хеш и число строк на узле должны совпасть с локальными.
Альтернатива — `make distill` на узле (`TEACHER_*` + сеть); меньше 2–3 тыс. пар даст переобучение за первую эпоху.

## 4. Порядок: dry-run → дым → три прогона → выгрузка
### 4.1. dry-run
Сначала `qat.train --dry-run` — 3 шага на случайных данных. Потом план раннера: без torch и без обучения он считает
шаги, часы и ₽ (проверено 2026-11-18: exit 0, каталоги не создаются).
```bash
mkdir -p runs && python3 -m qat.train --dry-run 2>&1 | tee runs/dry.log
python3 tools/embedding_experiment.py --data distill/out/pairs.jsonl --out runs/emb --dry-run
```
### 4.2. Один короткий прогон на настоящих данных
```bash
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/pairs.jsonl \
  --out runs/smoke --max-steps 20 2>&1 | tee runs/smoke.log
jq -r '[.step, .loss, .step_sec, .gpu_peak_gb] | @tsv' runs/smoke/log.jsonl | tail -3
```
Ожидаемо: 20 шагов без исключений, `gpu_peak_gb` < 24 ГБ, `step_sec` — секунда на шаг для §5. `--max-steps`
перекрывает `train.max_steps`; `--dry-run` этот флаг игнорирует (всегда 3 шага).
### 4.3. Три прогона
```bash
tmux new -s emb
python3 tools/embedding_experiment.py --data distill/out/pairs.jsonl --out runs/emb --epochs 3 2>&1 | tee -a runs/emb.log
```
Прогоны лягут в `runs/emb/none|int8|trit`, конфиги с подменённым `quantize_embeddings` — в `_configs/`, логи — в
`_logs/`. `--resume` пропускает готовые (DONE) и продолжает недоделанные; `--keep-going` — не бросать остальные
режимы, если один упал.
### 4.4. Экспорт и выгрузка
Раннер сам зовёт `qat.export` после каждого режима (отключается `--no-export`).
```bash
ls runs/emb/*/export/manifest.json; du -sh runs/emb/*/export
mkdir -p results-emb
rsync -avP "user@HOST:~/eva32/runs/emb/" results-emb/runs/
sha256sum results-emb/runs/*/export/ternary.trit; date -u
```
Метрика eval попадает в таблицу только через `--eval-cmd` (шаблон с `{mode}`, `{run}`, `{export}`, `{out}`,
`{eval_json}`) или готовым `runs/emb/eval-<режим>.json`; иначе колонка «—». Eval идёт дома, на поднятой модели:
`python3 eval/run_eval.py --model eva32 --repeat 3` против `EVA_BASE` (ключ — `VLLM_API_KEY`), в том числе на
`eval/hybrid_golden.jsonl`; чем поднимать экспорт — `docs/qat.md` §8. Инстанс выключать **после** rsync: сводная
таблица сравнения — `runs/emb/summary.json`, она и есть результат эксперимента (входит в тот же rsync).

## 5. Сколько это стоит
`₽ = часы × ставка + диск`; ставка — из панели, ориентир 4090 — 90–120 ₽/ч (`docs/distillation.md`), в раннере
`--rub-per-hour 100`. Часы = `шаги × step_sec / 3600` + 0,5 ч подготовки + 0,5 ч экспорта на прогон, где `шаги =
пары × эпохи / (micro_batch × grad_accum)`; `step_sec` — из §4.2, по умолчанию в раннере 10,8 с (2000 шагов ≈ 6 ч).
Таблица — на 10 тыс. пар; на 206 локальных парах раннер покажет 20 шагов и совсем другие часы.

| Сценарий | Шагов на прогон | Часов счёта (3 прогона) | Часов с арендой | ₽ при 100 ₽/ч | ₽ при 90–120 ₽/ч |
|---|---|---|---|---|---|
| **быстрый**: 1 эпоха | ~313 | 2,8 | 4,8 | 480 | 432–576 |
| **полный**: 3 эпохи | ~938 | 8,4 | 10,4 | 1 040 | 936–1 248 |
| «оставил на сутки» | ~938 | 8,4 | 24 | 2 400 | 2 160–2 880 |

Сутки вместо полного — +13,6 ч простоя и ≈ +1 360 ₽. Свои числа: `--sec-per-step` по замеру §4.2, ставка —
`--rub-per-hour`; тот же план печатает `--dry-run`.

## 6. Чек-лист перед тем, как платить за сутки
- [ ] `nvidia-smi` — нужная карта и ≥ 23 ГБ свободных; `df -h .` — ≥ 80 ГБ; `torch.cuda.is_available()` → `True`.
- [ ] `qat.train --dry-run` и `embedding_experiment.py --dry-run` прошли, план (часы, ₽) сходится с замером.
- [ ] `sha256sum distill/out/pairs.jsonl` на узле = локальному; пар ≈ 10 тыс., иначе 3 эпохи бессмысленны.
- [ ] дым-прогон 20 шагов без OOM: `gpu_peak_gb` известен, `step_sec` записан.
- [ ] прогон внутри `tmux`; чекпоинты `save_every: 100`, `keep_last_checkpoints: 2`.
- [ ] ставка и биллинг из панели записаны в отчёт; `date -u` зафиксирован до старта.

## 7. Если OOM
| Симптом | Причина | Что делать |
|---|---|---|
| OOM на первом шаге | micro-batch × seq не влезают в 24 ГБ | `micro_batch_size: 2` (затем 1), `max_seq_len: 1024`; gradient checkpointing уже включён |
| OOM в середине прогона | выросла длина / копия при сохранении | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (⚠️ сверить по `pip show torch`) |
| 21–22 ГБ занято, до OOM близко | `adam8bit` не подхватился, работает AdamW | `python3 -c "import bitsandbytes"`, `pip show bitsandbytes` |

```bash
cp qat/configs/qwen1.5b.yaml qat/configs/emb-oom.yaml
$EDITOR qat/configs/emb-oom.yaml
python3 -m qat.train --config qat/configs/emb-oom.yaml --data distill/out/pairs.jsonl --out runs/oom-check --max-steps 20
```

Крутить только в копии конфига — baseline `qwen1.5b.yaml` остаётся тем, что сравнивается; `gpu_peak_gb` в логе
подтверждает, стало ли легче.

## 8. Если прогон прервался
`qat.train` по SIGINT/SIGTERM завершает после текущего шага и сохраняет чекпоинт (`qat/train.py`), поэтому сигнал
лучше `kill -9`; чекпоинты — `runs/<режим>/checkpoints/step_000100`. Продолжение — тот же `--out`, `--config`,
`--data` и `--resume` без значения; `--allow-data-change` — только если файл данных менялся.
```bash
tmux new -s emb
python3 tools/embedding_experiment.py --out runs/emb --resume
python3 -m qat.train --config qat/configs/qwen1.5b.yaml --data distill/out/pairs.jsonl \
  --out runs/emb/trit --resume 2>&1 | tee -a runs/emb/trit/train.log   # если чинить один режим
jq -r '.latest_step, .status' runs/emb/trit/run.json
```
Ожидаемо: нумерация в `log.jsonl` продолжается, `latest_step` в `run.json` растёт, `loss` не прыгает вверх на
порядок. От смерти инстанса не спасают ни tmux, ни nohup — только чекпоинты, выкачанные к себе.

## 9. Что сверить
- CUDA-тег колеса torch под драйвер узла — `pip show torch`, `nvidia-smi` (§2).
- `HOST`/пользователь/порт/путь — панель провайдера (§3).
- `--eval-cmd` и раскладку `runs/emb/none|int8|trit` — шаблон в коде есть, прогоном не проверен (§4.4).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` на этой сборке torch (§7).
- что `--resume` восстанавливает эпоху и оптимизатор — `run.json`, хвост `log.jsonl` (§8).
