#!/bin/sh
# Запуск сервера Евы из ТЕРМИНАЛА пользователя.
#
# Зачем скрипт: разрешения macOS (автоматизация Music, микрофон) выдаются
# приложению, из которого запущен сервер. Если сервер поднят из среды
# разработки, macOS спрашивает «DeepSeek», и музыка/запись работают только
# пока жива та среда. Запуск из Терминала закрепляет права за Терминалом.
set -e
cd "$(dirname "$0")/.."
pkill -f "tools/chat_server.py" 2>/dev/null || true
sleep 1
nohup python3 tools/chat_server.py --port 8765 > /tmp/cs.log 2>&1 &
sleep 6
if curl -s --max-time 3 http://127.0.0.1:8765/api/models >/dev/null; then
    echo "сервер поднят: http://127.0.0.1:8765 (лог /tmp/cs.log)"
else
    echo "сервер не ответил — смотрите /tmp/cs.log" >&2
    exit 1
fi
