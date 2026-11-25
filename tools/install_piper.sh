#!/bin/sh
# Установка локального нейро-TTS Piper в vendor/ (в git не попадает: .gitignore).
#
# Что делает:
#   1) ставит Python-пакет piper-tts в vendor/piper-py (нужен python3.13 с pip:
#      у него есть колёса на arm64; python3.14 колёс ещё не имеет);
#   2) качает русский голос ru_RU-irina-medium (onnx ~60 МБ) в vendor/piper/voices;
#   3) кладёт обёртку vendor/piper/bin/piper, которую ждёт eva/tts.py
#      (`piper --model M --output_file OUT`, текст в stdin).
#
# После установки чат подхватывает голос сам, если задать:
#   PIPER_BIN=vendor/piper/bin/piper \
#   PIPER_MODEL=vendor/piper/voices/ru_RU-irina-medium.onnx
# (или положить эти значения в окружение запуска chat_server.py).
#
# Задача: 0183. Сеть нужна только здесь; сам синтез идёт офлайн.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

py=${PIPER_PYTHON:-/opt/homebrew/bin/python3.13}
voice_base=https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium

mkdir -p vendor/piper/bin vendor/piper/voices

if [ ! -d vendor/piper-py/piper ]; then
  echo "== ставлю piper-tts в vendor/piper-py"
  "$py" -m pip install -q --disable-pip-version-check --target vendor/piper-py piper-tts
fi

for f in ru_RU-irina-medium.onnx ru_RU-irina-medium.onnx.json; do
  if [ ! -s "vendor/piper/voices/$f" ]; then
    echo "== качаю $f"
    curl -sL -o "vendor/piper/voices/$f" "$voice_base/$f"
  fi
done

cat > vendor/piper/bin/piper <<'WRAP'
#!/bin/sh
# Обёртка над Python-пакетом piper-tts: eva/tts.py зовёт `piper --model M
# --output_file OUT` с текстом на stdin, а у нас piper установлен как модуль
# в vendor/piper-py (onnxruntime + голоса). Пути вычисляются от самого скрипта.
here=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$here/../piper-py" exec "${PIPER_PYTHON:-/opt/homebrew/bin/python3.13}" -m piper "$@"
WRAP
chmod +x vendor/piper/bin/piper

echo "== проверка синтеза"
PIPER_BIN=vendor/piper/bin/piper PIPER_MODEL=vendor/piper/voices/ru_RU-irina-medium.onnx \
  python3 -c "
import sys; sys.path.insert(0, '.')
from eva.tts import synthesize
w = synthesize('Проверка голоса')
data = w if isinstance(w, bytes) else open(w, 'rb').read()
assert data[:4] == b'RIFF', data[:16]
print('ok:', len(data), 'байт WAV')
"
