"""``zeroproof`` command line: login, signup, logout, status."""

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
    p_login.add_argument(
        "--no-wait", action="store_true", help="return after printing the link; run again to finish"
    )
    p_login.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds to wait for approval (default: until the code expires)",
    )

    p_signup = sub.add_parser("signup", help="create an account and a key, no browser")
    p_signup.add_argument("--email", required=True, help="address for the new account")
    p_signup.add_argument("--name", help="name for the key on the account (default: cli <host>)")

    sub.add_parser("logout", help="delete the saved key")
    sub.add_parser("status", help="show which key the SDK will use")

    p_purge = sub.add_parser(
        "purge", help="delete an agent's traces, datasets and record, or empty datasets"
    )
    p_purge.add_argument("--agent", help="agent slug to remove with everything under it")
    p_purge.add_argument(
        "--empty", action="store_true", help="delete datasets with no stored bytes"
    )
    p_purge.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="with --empty: also delete sets with this many rows or fewer",
    )
    p_purge.add_argument("--dry-run", action="store_true", help="count, delete nothing")
    p_purge.add_argument("--yes", action="store_true", help="skip the confirmation")

    args = parser.parse_args(argv)

    if args.command == "login":
        try:
            key = auth.login(
                name=args.name,
                wait=not args.no_wait,
                timeout=args.timeout,
                open_browser=not args.no_browser,
            )
        except auth.LoginError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        return 0 if key or args.no_wait else 2

    if args.command == "signup":
        try:
            auth.signup(args.email, name=args.name)
        except auth.LoginError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        return 0

    if args.command == "logout":
        print("Logged out." if auth.logout() else "No saved key.")
        return 0

    if args.command == "status":
        shown = auth.status()
        print(json.dumps(shown, indent=2))
        if not shown.get("configured"):
            print(
                "no API key configured: run `zeroproof login` or set ZEROPROOF_API_KEY",
                file=sys.stderr,
            )
        return 0

    if args.command == "purge":
        from .simulations.ingest.platform import delete_empty_datasets, purge_agent

        if not args.agent and not args.empty:
            print("error: pass --agent <slug> and/or --empty", file=sys.stderr)
            return 1
        plan: dict = {}
        if args.agent:
            plan["agent"] = purge_agent(args.agent, dry_run=True)
        if args.empty:
            plan["empty"] = delete_empty_datasets(max_rows=args.max_rows, dry_run=True)
        print(json.dumps(plan, indent=2))
        if args.dry_run:
            return 0
        if not args.yes:
            answer = input("Delete all of the above? This cannot be undone. [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Nothing deleted.")
                return 2
        done: dict = {}
        if args.agent:
            done["agent"] = purge_agent(args.agent)
        if args.empty:
            done["empty"] = delete_empty_datasets(max_rows=args.max_rows)
        print(json.dumps(done, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
