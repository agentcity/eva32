"""CLI гибрида: `ask` для одного вопроса, `serve` для сервера.

Контракт упоминает `python3 -m eva.cli ask "..." --explain` (docs/hybrid.md,
runbook), поэтому вход должен существовать, а не только `python3 -m eva`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import load
from .router import answer
from .types import Message, Request


def _request(text: str, user_id: str | None) -> Request:
    meta = {"channel": "cli"}
    if user_id:
        meta["user_id"] = user_id
    return Request(messages=[Message(role="user", content=text)], meta=meta)


def cmd_ask(args: argparse.Namespace) -> int:
    from .policy import BudgetExceeded, ModelUnavailable

    cfg = load()
    try:
        result = answer(_request(args.text, args.user_id), cfg)
    except (BudgetExceeded, ModelUnavailable) as exc:
        print(f"не могу ответить: {exc}", file=sys.stderr)
        if args.explain:
            print("маршрут требует модель, а она не настроена "
                  "(EVA_BIG_BASE/EVA_BIG_KEY или локальная EVA_SMALL_BASE)", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI не должен показывать трейсбек
        print(f"ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(result.text)
    if args.explain:
        route = result.route
        print(file=sys.stderr)
        print(f"маршрут:  {route.kind} ({route.confidence:.2f})", file=sys.stderr)
        print(f"причина:  {route.reason}", file=sys.stderr)
        print(f"скилл:    {result.skill or '—'}", file=sys.stderr)
        print(f"цена:     {result.cost_rub:.4f} ₽", file=sys.stderr)
        print(f"латентн.: {result.latency_ms} мс", file=sys.stderr)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import create_server

    server = create_server(host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"eva-server: http://{host}:{port}", flush=True)
    if not os.environ.get("EVA_AUTH_TOKEN"):
        print("внимание: EVA_AUTH_TOKEN не задан — только для localhost", file=sys.stderr)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eva.cli", description="гибридный ассистент EVA32")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="задать один вопрос")
    ask.add_argument("text", help="текст вопроса")
    ask.add_argument("--explain", action="store_true", help="показать маршрут, причину и цену")
    ask.add_argument("--json", action="store_true", help="вывести ответ как JSON")
    ask.add_argument("--user-id", default=None, help="идентификатор пользователя для телеметрии")
    ask.set_defaults(func=cmd_ask)

    serve = sub.add_parser("serve", help="поднять OpenAI-совместимый сервер")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
