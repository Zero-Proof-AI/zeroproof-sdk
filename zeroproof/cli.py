"""``zeroproof`` command line: login, logout, status."""

from __future__ import annotations

import argparse
import json
import sys

from . import auth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeroproof", description="Zero Proof Labs SDK")
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="sign in from this terminal (opens the browser)")
    p_login.add_argument("--name", help="name for the key on your account (default: cli <host>)")
    p_login.add_argument("--no-browser", action="store_true", help="print the link only")
    p_login.add_argument("--no-wait", action="store_true",
                         help="return after printing the link; run again to finish")
    p_login.add_argument("--timeout", type=float, default=None,
                         help="seconds to wait for approval (default: until the code expires)")

    sub.add_parser("logout", help="delete the saved key")
    sub.add_parser("status", help="show which key the SDK will use")

    args = parser.parse_args(argv)

    if args.command == "login":
        try:
            key = auth.login(name=args.name, wait=not args.no_wait, timeout=args.timeout,
                             open_browser=not args.no_browser)
        except auth.LoginError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        return 0 if key or args.no_wait else 2

    if args.command == "logout":
        print("Logged out." if auth.logout() else "No saved key.")
        return 0

    if args.command == "status":
        print(json.dumps(auth.status(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
