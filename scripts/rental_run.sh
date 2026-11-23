#!/usr/bin/env bash
#
# Один прогон QAT на арендованной GPU — по шагам, с остановкой после val loss.
#
#   шаг 1: preflight окружения (tools/rental_preflight.py) — «можно/нельзя»
#   шаг 2: qat.train --dry-run — оценка памяти и 3 шага на случайных данных
#   шаг 3: STEPS шагов обучения (по умолчанию 100) на настоящих данных
#   шаг 4: val loss по отложенной выборке (tools/eval_torch.py → perplexity)
#
# Дальше скрипт останавливается: полный пилот (350 шагов) и экспорт — отдельными
# командами, которые он печатает в конце. Всё пишется в runs/rental/.
#
# Ничего не устанавливает, не скачивает и не ходит в сеть: только проверки и
# запуск уже установленного стека. pip — отдельно, на узле (рунбук §2).
#
# Примеры:
#   bash scripts/rental_run.sh --dry-run              # план без обучения
#   bash scripts/rental_run.sh                        # 100 шагов + val loss
#   bash scripts/rental_run.sh --steps 350            # полный пилот
#   bash scripts/rental_run.sh --data runs/mixture/train.jsonl --val runs/mixture/val.jsonl
#
# Коды возврата: 0 — успех; 1 — preflight/обучение/val не прошли; 2 — аргументы.
# Смотреть: runs/rental/<id>.log, runs/rental/<id>/step<N>/log.jsonl, val.json.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

CONFIG="qat/configs/qwen1.5b.yaml"
DATA=""
VAL=""
STEPS="100"
SEC_PER_STEP="10.8"
RUB_MIN="90"
RUB_MAX="120"
PP_LIMIT="100"
MAX_NEW_TOKENS="8"
RUN_ROOT="runs/rental"
NAME=""
PLAN_ONLY="0"

usage() {
  cat <<'EOF'
rental_run.sh — дым-прогон QAT 1.5B на арендованной GPU.

Опции:
  --dry-run            показать план и прогнать проверки, обучение не запускать
  --steps N            сколько шагов обучать (по умолчанию 100; полный пилот — 350)
  --config PATH        QAT-конфиг (по умолчанию qat/configs/qwen1.5b.yaml)
  --data PATH          JSONL с парами (по умолчанию runs/mixture/train.jsonl —
                       наш микс 75 тыс., затем data/train.jsonl — split 0092,
                       затем distill/out/code_pairs.jsonl)
  --val PATH           отложенная выборка для val loss (по умолчанию val.jsonl
                       рядом с train: data/train.jsonl → data/val.jsonl)
  --pp-limit N         сколько пар брать для val loss (по умолчанию 100)
  --sec-per-step S     оценка секунд на шаг для бюджета (по умолчанию 10.8, 0036)
  --rub-min N          нижняя ставка ₽/ч для оценки (по умолчанию 90)
  --rub-max N          верхняя ставка ₽/ч для оценки (по умолчанию 120)
  --name ID            имя прогона (по умолчанию UTC-метка времени)
  -h, --help           эта справка

Результат: runs/rental/<id>.log и runs/rental/<id>/ (dryrun/, step<N>/, val.json).
EOF
}

die() {
  echo "rental_run: $*" >&2
  echo "справка: bash scripts/rental_run.sh --help" >&2
  exit 2
}

need_value() {
  [ "$#" -ge 2 ] || die "опция $1 требует значение"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) PLAN_ONLY="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    --steps) need_value "$@"; STEPS="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --data) need_value "$@"; DATA="$2"; shift 2 ;;
    --val) need_value "$@"; VAL="$2"; shift 2 ;;
    --pp-limit) need_value "$@"; PP_LIMIT="$2"; shift 2 ;;
    --sec-per-step) need_value "$@"; SEC_PER_STEP="$2"; shift 2 ;;
    --rub-min) need_value "$@"; RUB_MIN="$2"; shift 2 ;;
    --rub-max) need_value "$@"; RUB_MAX="$2"; shift 2 ;;
    --name) need_value "$@"; NAME="$2"; shift 2 ;;
    *) die "неизвестная опция: $1" ;;
  esac
done

case "$STEPS" in
  ''|*[!0-9]*) die "--steps: нужно целое число, получено '$STEPS'" ;;
esac
[ "$STEPS" -gt 0 ] || die "--steps: нужно число больше нуля"
case "$PP_LIMIT" in
  ''|*[!0-9]*) die "--pp-limit: нужно целое число, получено '$PP_LIMIT'" ;;
esac
case "$SEC_PER_STEP" in
  ''|*[!0-9.]*) die "--sec-per-step: нужно число, получено '$SEC_PER_STEP'" ;;
esac
case "$RUB_MIN" in
  ''|*[!0-9]*) die "--rub-min: нужно целое число" ;;
esac
case "$RUB_MAX" in
  ''|*[!0-9]*) die "--rub-max: нужно целое число" ;;
esac

[ -f "$CONFIG" ] || die "конфиг не найден: $CONFIG (запуск из корня репозитория)"
command -v python3 >/dev/null 2>&1 || die "нет python3 в PATH"

# --- данные: то же, что ищет preflight -------------------------------------
if [ -z "$DATA" ]; then
  if [ -f "runs/mixture/train.jsonl" ]; then
    DATA="runs/mixture/train.jsonl"
  elif [ -f "data/train.jsonl" ]; then
    DATA="runs/mixture/train.jsonl"
  elif [ -f "distill/out/code_pairs.jsonl" ]; then
    DATA="distill/out/code_pairs.jsonl"
  else
    for candidate in distill/out/*.jsonl; do
      if [ -f "$candidate" ]; then
        DATA="$candidate"
        break
      fi
    done
  fi
fi
[ -n "$DATA" ] || die "нет данных: data/train.jsonl, runs/mixture/train.jsonl и distill/out/*.jsonl"
[ -f "$DATA" ] || die "файл данных не найден: $DATA"

if [ -z "$VAL" ] && [ -f "$(dirname "$DATA")/val.jsonl" ]; then
  VAL="$(dirname "$DATA")/val.jsonl"
fi
if [ -n "$VAL" ] && [ ! -f "$VAL" ]; then
  die "файл val не найден: $VAL"
fi

# --- прогон и лог ----------------------------------------------------------
if [ -n "$NAME" ]; then
  RUN_ID="$NAME"
else
  RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')"
fi
case "$RUN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "--name: только буквы, цифры, точка, '_' и '-'" ;;
esac
RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_ROOT" "$RUN_DIR"
LOG="$RUN_ROOT/$RUN_ID.log"
STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# Ловим падение любой команды и объясняем OOM; exit-код сохраняется.
on_error() {
  local code=$?
  trap - ERR
  echo ""
  echo "=== СТОП: команда завершилась с кодом $code ==="
  # Даём tee дописать хвост вывода в лог, иначе OOM-строки может ещё не быть.
  sleep 1
  if grep -qiE 'out of memory|CUDA OOM|CudaErrorMemoryAllocation' "$LOG" 2>/dev/null; then
    echo "Похоже на OOM (рунбук §12):"
    echo "  • уменьшайте micro_batch_size (4 → 2 → 1) в КОПИИ конфига:"
    echo "      cp qat/configs/qwen1.5b.yaml qat/configs/oom.yaml && \$EDITOR qat/configs/oom.yaml"
    echo "  • PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True при запуске"
    echo "  • проверьте, что Adam8bit подхватился: python3 -c \"import bitsandbytes\""
  fi
  echo "лог: $LOG"
  echo "каталог: $RUN_DIR"
  exit "$code"
}

main() {
  trap on_error ERR

  # --- бюджет: счёт + 0.5 ч подготовки + ~0.25 ч val ----------------------
  local budget count_min
  budget="$(awk -v s="$STEPS" -v sec="$SEC_PER_STEP" -v lo="$RUB_MIN" -v hi="$RUB_MAX" \
    'BEGIN { h = s * sec / 3600 + 0.75; printf "%.2f %.0f %.0f\n", h, h * lo, h * hi }')"
  local BUDGET_HOURS BUDGET_RUB_LO BUDGET_RUB_HI
  read -r BUDGET_HOURS BUDGET_RUB_LO BUDGET_RUB_HI <<<"$budget"
  count_min="$(awk -v s="$STEPS" -v sec="$SEC_PER_STEP" 'BEGIN { printf "%.0f", s * sec / 60 }')"

  echo "=== rental_run: QAT 1.5B на арендованной GPU ==="
  echo "старт:    $STARTED_AT (UTC)"
  echo "каталог:  $ROOT"
  echo "конфиг:   $CONFIG"
  echo "данные:   $DATA"
  if [ -n "$VAL" ]; then
    echo "val:      $VAL"
  else
    echo "val:      нет — val loss считать не на чем (шаг 4 будет пропущен)"
  fi
  echo "шагов:    $STEPS (полный пилот из конфига — 350)"
  echo "оценка:   $count_min мин счёта + подготовка/dry-run/val ≈ $BUDGET_HOURS ч" \
    "≈ ${BUDGET_RUB_LO}–${BUDGET_RUB_HI} ₽"
  echo "полный пилот: 350 шагов ≈ 2,1 ч ≈ 192–256 ₽ при 90–120 ₽/ч (рунбук §10)"
  echo "лог:      $LOG"
  echo "каталог:  $RUN_DIR"

  # --- шаг 1: preflight ----------------------------------------------------
  echo ""
  echo "--- шаг 1/4: preflight окружения ---"
  local ok=1
  if [ -n "$VAL" ]; then
    if ! python3 tools/rental_preflight.py --config "$CONFIG" --data "$DATA" --val "$VAL"; then
      ok=0
    fi
  else
    if ! python3 tools/rental_preflight.py --config "$CONFIG" --data "$DATA"; then
      ok=0
    fi
  fi

  if [ "$ok" = "0" ]; then
    echo ""
    echo "preflight: НЕЛЬЗЯ (список [FAIL] выше)."
    if [ "$PLAN_ONLY" = "1" ]; then
      echo "режим --dry-run: продолжаю только проверки и план, обучение не запускаю."
    else
      echo "Обучение не запускаю: час аренды ушёл бы на те же ошибки."
      echo "Исправьте [FAIL] и повторите: bash scripts/rental_run.sh"
      exit 1
    fi
  fi

  # --- шаг 2: dry-run памяти и импортов -----------------------------------
  echo ""
  echo "--- шаг 2/4: qat.train --dry-run (оценка памяти + 3 шага на случайных данных) ---"
  if ! python3 -m qat.train --config "$CONFIG" --dry-run --out "$RUN_DIR/dryrun"; then
    echo ""
    if [ "$PLAN_ONLY" = "1" ]; then
      echo "ВНИМАНИЕ: dry-run не прошёл (см. вывод выше), но --dry-run только печатает план."
      echo "На GPU-узле это [FAIL]: без него обучение не стартует."
    else
      echo "dry-run не прошёл — обучение не запускаю (импорты/CUDA/конфиг)."
      exit 1
    fi
  fi

  if [ "$PLAN_ONLY" = "1" ]; then
    echo ""
    echo "=== --dry-run: план проверен, обучение не запускалось ==="
    echo "следующая команда на GPU-узле:"
    echo "  bash scripts/rental_run.sh --config $CONFIG --data $DATA${VAL:+ --val $VAL}"
    echo "полный пилот: bash scripts/rental_run.sh --steps 350 (≈2,1 ч ≈192–256 ₽)"
    exit 0
  fi

  # --- шаг 3: обучение ----------------------------------------------------
  echo ""
  echo "--- шаг 3/4: $STEPS шагов обучения ---"
  echo "прогресс: tail -f $RUN_DIR/step$STEPS/log.jsonl"
  python3 -m qat.train --config "$CONFIG" --data "$DATA" \
    --out "$RUN_DIR/step$STEPS" --max-steps "$STEPS" 2>&1 | tee "$RUN_DIR/train.log"

  echo ""
  echo "последние шаги из $RUN_DIR/step$STEPS/log.jsonl:"
  python3 - "$RUN_DIR/step$STEPS/log.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("  log.jsonl нет — обучение не писало лог")
    sys.exit(0)
rows = []
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue
for row in rows[-3:]:
    print(
        f"  шаг {row.get('step')}: loss {row.get('loss')} "
        f"ce {row.get('ce')} · {row.get('step_sec')} с/шаг"
    )
PY

  # --- шаг 4: val loss ----------------------------------------------------
  echo ""
  if [ -z "$VAL" ]; then
    echo "--- шаг 4/4: val loss пропущен (нет val-файла) ---"
    echo "val loss без отложенной выборки не измерить; передайте --val PATH"
  else
    echo "--- шаг 4/4: val loss на $VAL (perplexity по токенам ответа) ---"
    echo "это ~$PP_LIMIT пар, несколько минут; golden — 1 вопрос для дым-проверки"
    python3 tools/eval_torch.py \
      --model "$RUN_DIR/step$STEPS" \
      --golden eval/golden.jsonl \
      --ppl-data "$VAL" \
      --ppl-limit "$PP_LIMIT" \
      --limit 1 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --local-files-only \
      --out "$RUN_DIR/val.json"

    echo ""
    python3 - "$RUN_DIR/val.json" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"val loss не прочитан: {exc}")
    sys.exit(1)
perplexity = data.get("perplexity")
if perplexity is None:
    print("val loss: perplexity null — на val-файле не нашлось пар для замера")
    sys.exit(0)
pairs = data.get("ppl_pairs")
tokens = data.get("ppl_tokens")
print(
    f"val loss = {math.log(float(perplexity)):.4f} "
    f"(perplexity {float(perplexity):.3f} на {pairs} парах, {tokens} токенов)"
)
print("сравнивать с base-моделью на тех же парах, а не с train loss")
PY
  fi

  # --- итог и следующие шаги ----------------------------------------------
  echo ""
  echo "=== готово: $STEPS шагов ==="
  echo "лог:      $LOG"
  echo "прогон:   $RUN_DIR/step$STEPS"
  if [ -n "$VAL" ]; then
    echo "val:      $RUN_DIR/val.json"
  fi
  echo ""
  echo "дальше — по рунбуку §0.1:"
  echo "  полный пилот: bash scripts/rental_run.sh --steps 350 --name pilot1"
  echo "  экспорт:      python3 -m qat.export --run $RUN_DIR/step$STEPS --out export/rental"
  echo "  выгрузка:     rsync -avP \"user@HOST:~/eva32/runs/rental/\" runs/rental/"
  echo "  и выключить инстанс сразу после rsync (рунбук §10)"
}

# Весь вывод идёт и в терминал, и в лог; pipefail сохраняет код возврата main.
main "$@" 2>&1 | tee -a "$LOG"
