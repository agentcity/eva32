#!/bin/sh
# Установка локального нейро-TTS Silero в vendor/ (в git не попадает: .gitignore).
#
# Что делает:
#   1) проверяет python3.13 (у него есть колёса torch для macOS arm64;
#      у python3.14 их нет);
#   2) ставит torch в vendor/torch-py (в .pylibs лежит сломанный
#      source-checkout — его не трогаем и не используем);
#   3) качает модель Silero v4_ru через torch.hub в vendor/silero
#      (TORCH_HOME=vendor/silero; первый раз нужна сеть);
#   4) делает пробный синтез «Проверка голоса» и печатает размеры/время.
#
# Повторный запуск безопасен: torch проверяется импортом, модель берётся из
# кэша, пробный синтез просто перезаписывает vendor/silero/check.wav.
#
# После установки чат подхватывает Silero сам (EVA_TTS=auto): порядок
# ElevenLabs → Silero → Piper → system → null. Явно:
#   EVA_TTS=silero SILERO_SPEAKER=baya python3 tools/chat_server.py --port 8765
#
# Задача: 0193. Сеть нужна только здесь; сам синтез идёт офлайн.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

py=${SILERO_PYTHON:-/opt/homebrew/bin/python3.13}
torch_dir=vendor/torch-py
model_dir=vendor/silero
speaker=${SILERO_SPEAKER:-baya}
check_wav="$model_dir/check.wav"

if [ ! -x "$py" ]; then
  py=$(command -v python3.13 || true)
fi
if [ -z "$py" ] || [ ! -x "$py" ]; then
  echo "!! нужен python3.13 с pip (в системе: /opt/homebrew/bin/python3.13)." >&2
  echo "   python3.14 не подходит: для него нет колёс torch на macOS arm64." >&2
  exit 1
fi
if ! "$py" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)'; then
  echo "!! $py — не Python 3.13; нужен именно 3.13." >&2
  exit 1
fi

mkdir -p "$torch_dir" "$model_dir"

if PYTHONPATH="$torch_dir" "$py" -c 'import torch, omegaconf' >/dev/null 2>&1; then
  torch_version=$(
    PYTHONPATH="$torch_dir" "$py" -c 'import torch; print(torch.__version__)'
  )
  echo "== torch уже установлен: $torch_version ($torch_dir)"
else
  echo "== ставлю torch в $torch_dir (скачивание, нужна сеть)"
  # omegaconf — зависимость silero-models (models.yml), не только torch.
  "$py" -m pip install --no-cache-dir --disable-pip-version-check \
    --target "$torch_dir" torch numpy omegaconf
  torch_version=$(
    PYTHONPATH="$torch_dir" "$py" -c 'import torch; print(torch.__version__)'
  )
  echo "== torch установлен: $torch_version"
fi

echo "== загружаю модель Silero v4_ru и синтезирую «Проверка голоса»"
trial=$(
  TORCH_HOME="$model_dir" \
  PYTHONPATH="$torch_dir" \
  SILERO_SPEAKER="$speaker" \
  "$py" tools/silero_speak.py "$check_wav" <<'TXT'
Проверка голоса
TXT
)
printf '%s\n' "$trial"

echo "== размеры"
du -sh "$torch_dir" "$model_dir" 2>/dev/null || true
ls -lh "$check_wav" 2>/dev/null | awk '{print "check.wav:", $5}'
find "$model_dir/hub/checkpoints" \
  "$model_dir/hub/snakers4_silero-models_master/src/silero/model" \
  -maxdepth 1 -type f -name '*.pt' 2>/dev/null \
  -exec ls -lh {} \; | awk '{print "weights:", $5, $9}'

echo "== готово"
echo "   EVA_TTS=auto → Silero (голос $speaker) → Piper → system → null"
echo "   сменить голос: SILERO_SPEAKER=kseniya|baya|xenia|aidar|eugene"
