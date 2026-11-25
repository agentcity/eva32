#!/bin/sh
# Сборка EVA.app — лёгкого нативного приложения со статус-баром, голосовым
# вводом (Apple Speech, ru-RU, on-device) и озвучкой ответов через /api/tts.
#
# Задача: 0184. Зависимостей нет: только swiftc из Xcode Command Line Tools.
#
# Использование:
#   sh tools/build_eva_app.sh              # собрать build/EVA.app
#   sh tools/build_eva_app.sh --run        # собрать и запустить
#   sh tools/build_eva_app.sh --clean      # удалить build/EVA.app
#   EVA_OUT_DIR=/tmp/eva sh tools/build_eva_app.sh
#
# Почему так, а не Xcode-проектом:
#   * один swiftc-вызов собирает все .swift, никаких пакетов и зависимостей;
#   * модульный кэш уводится в /tmp: в репозитории swiftc падает с
#     «Operation not permitted» на кэше модулей (подтверждено в задаче);
#   * Info.plist и ad-hoc подпись обязательны: без .app-бандла Apple Speech не
#     даёт авторизацию, а без подписи на Apple Silicon бинарь не запустится.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
src="$root/apps/eva/Sources"
out_dir="${EVA_OUT_DIR:-$root/build}"
app="$out_dir/EVA.app"
plist_src="$root/apps/eva/Info.plist"

clean=0
run=0
for arg in "$@"; do
  case "$arg" in
    --clean) clean=1 ;;
    --run|-r) run=1 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "неизвестный аргумент: $arg (см. --help)" >&2
      exit 2
      ;;
  esac
done

if [ "$clean" = 1 ]; then
  rm -rf "$app"
  echo "удалено: $app"
  [ -d "$src" ] || exit 0
fi

command -v swiftc >/dev/null 2>&1 || {
  echo "нет swiftc: установите Xcode Command Line Tools (xcode-select --install)" >&2
  exit 1
}
[ -f "$plist_src" ] || { echo "нет $plist_src" >&2; exit 1; }
[ -d "$src" ] || { echo "нет каталога исходников $src" >&2; exit 1; }

arch=$(uname -m)
case "$arch" in
  arm64) target="arm64-apple-macosx13.0" ;;
  x86_64) target="x86_64-apple-macosx13.0" ;;
  *) target="$(uname -m)-apple-macosx13.0" ;;
esac

# Модульный кэш — в /tmp: на этом репозитории swiftc иначе падает на правах.
SWIFT_MODULECACHE_PATH="${EVA_SWIFT_MODULECACHE:-/tmp/eva-app-swift-modulecache}"
CLANG_MODULE_CACHE_PATH="${EVA_CLANG_MODULECACHE:-/tmp/eva-app-clang-modulecache}"
export SWIFT_MODULECACHE_PATH CLANG_MODULE_CACHE_PATH
mkdir -p "$SWIFT_MODULECACHE_PATH" "$CLANG_MODULE_CACHE_PATH"

echo "== чищу $app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

echo "== swiftc ($target)"
# shellcheck disable=SC2086
swiftc \
  -O \
  -target "$target" \
  -swift-version 6 \
  -module-name EVA \
  -framework AppKit \
  -framework AVFoundation \
  -framework Carbon \
  -framework Speech \
  -o "$app/Contents/MacOS/EVA" \
  "$src"/*.swift

echo "== Info.plist"
cp "$plist_src" "$app/Contents/Info.plist"
chmod 644 "$app/Contents/Info.plist"
plutil -lint "$app/Contents/Info.plist" >/dev/null
printf 'APPL????' > "$app/Contents/PkgInfo"

echo "== ad-hoc подпись"
# Подпись нужна и для запуска на arm64, и для TCC: разрешения привязываются
# к бандлу. Каждая пересборка меняет cdhash, поэтому macOS может спросить
# микрофон/распознавание заново — это ожидаемо (см. docs/voice-app.md).
codesign --force --sign - "$app" >/dev/null
codesign --verify --strict --verbose=1 "$app" 2>&1 | sed 's/^/   /'

size_kb=$(du -sk "$app" | awk '{print $1}')
size_mb=$(awk -v kb="$size_kb" 'BEGIN { printf "%.1f", kb/1024 }')
echo "== готово: $app (${size_mb} МБ)"
if [ "$size_kb" -gt 20480 ]; then
  echo "ВНИМАНИЕ: приложение больше 20 МБ — критерий задачи 0184 нарушен" >&2
fi

echo "   запуск:      open \"$app\""
echo "   проверка:    \"$app/Contents/MacOS/EVA\" --selftest-chat \"привет\""
echo "   журнал:      ~/Library/Logs/EVA/eva.log"

if [ "$run" = 1 ]; then
  echo "== открываю"
  open "$app"
fi
