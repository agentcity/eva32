"""Запуск HTTP-сервера EVA32: ``python3 -m eva``.

Сервер — транспорт; боевой маршрутизатор (``eva.router.answer``) собирается
лениво при первом запросе к ``/v1/chat/completions``. Поэтому ``/health``
отвечает сразу, а неготовность моделей не мешает поднять процесс.

Примеры::

    python3 -m eva                       # 127.0.0.1:8000
    python3 -m eva --port 0              # свободный порт, печатается в stdout
    python3 -m eva --router my_router:build   # своя фабрика маршрутизатора

``--router`` принимает ``МОДУЛЬ:ФАБРИКА``, где фабрика без аргументов
возвращает функцию ``router(request) -> Answer`` (или объект с методом
``answer(request)``). Это нужно для стенда и отладки вместо боевой связки.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from .server import DEFAULT_MODEL, create_server, version

LOGGER_NAME = "eva.server"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eva",
        description="EVA32: OpenAI-совместимый сервер гибридного ассистента.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="адрес для прослушивания (по умолчанию 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="порт; 0 — взять свободный и напечатать его (по умолчанию 8000)",
    )
    parser.add_argument(
        "--router",
        metavar="МОДУЛЬ:ФАБРИКА",
        default=None,
        help="фабрика маршрутизатора без аргументов, например стенд:build",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="уровень журнала (по умолчанию INFO)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="eva-server {}".format(version()),
    )
    return parser.parse_args(argv)


def router_from_spec(spec: str):
    """Импортирует ``МОДУЛЬ:ФАБРИКА`` и вызывает фабрику без аргументов."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise SystemExit(
            "--router: ожидается МОДУЛЬ:ФАБРИКА, например стенд:build_router"
        )

    try:
        module = importlib.import_module(module_name.strip())
    except ImportError as exc:
        raise SystemExit(
            "--router: модуль {!r} не импортируется: {}".format(module_name, exc)
        ) from exc

    factory = getattr(module, attribute.strip(), None)
    if not callable(factory):
        raise SystemExit(
            "--router: в модуле {!r} нет вызываемого {!r}".format(
                module_name, attribute
            )
        )

    router = factory()
    if not callable(router) and not callable(getattr(router, "answer", None)):
        raise SystemExit(
            "--router: фабрика {!r} вернула не маршрутизатор".format(attribute)
        )
    return router


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    router = router_from_spec(args.router) if args.router else None

    try:
        httpd = create_server(host=args.host, port=args.port, router=router)
    except OSError as exc:
        reason = getattr(exc, "strerror", None) or str(exc)
        print(
            "eva-server: не удалось занять {}:{}: {}".format(
                args.host, args.port, reason
            ),
            file=sys.stderr,
        )
        return 1

    host, port = httpd.server_address[:2]
    print(
        "eva-server {}: http://{}:{} (модель {})".format(
            version(), host, port, DEFAULT_MODEL
        ),
        flush=True,
    )

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("eva-server: остановлен по Ctrl+C", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
