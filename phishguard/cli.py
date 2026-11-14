"""Command line entry point.

  python -m phishguard.cli check  email.eml [more.eml ...]     # score .eml files
  python -m phishguard.cli check  --subject "..." --body "..."  # score text
  python -m phishguard.cli serve  [--host 0.0.0.0 --port 8000]  # REST API + web UI
  python -m phishguard.cli connect imap|gmail|outlook [--interval 60] [--once] [--dry-run] [--move]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_dotenv


def _print_verdict(v: dict, name: str = "") -> None:
    tag = "PHISHING  " if v["is_phishing"] else "LEGITIMATE"
    print(f"{tag} score={v['score']:.3f} ({v['confidence']})  {name}")
    for r in v["reasons"]:
        print(f"    - {r}")


def cmd_check(args) -> None:
    from .detector import get_detector
    det = get_detector()
    if args.files:
        for f in args.files:
            v = det.predict_raw(Path(f).read_bytes()).to_dict()
            print(json.dumps(v, indent=2)) if args.json else _print_verdict(v, f)
    else:
        v = det.predict_fields(args.subject or "", args.body or sys.stdin.read(), args.sender or "").to_dict()
        print(json.dumps(v, indent=2)) if args.json else _print_verdict(v)


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("phishguard.api:app", host=args.host, port=args.port, workers=1)


def cmd_connect(args) -> None:
    kw = {"dry_run": args.dry_run, "tag_safe": not args.no_safe_tag}
    if args.move:
        kw["move_phishing"] = True
    if args.provider == "imap":
        from .connectors.imap_connector import IMAPConnector as C
    elif args.provider == "gmail":
        from .connectors.gmail_connector import GmailConnector as C
    else:
        from .connectors.outlook_connector import OutlookConnector as C
    conn = C(**kw)
    if args.once:
        conn.connect()
        print(conn.run_once())
    else:
        conn.run_forever(interval=args.interval)


def main(argv=None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="phishguard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="score email files or text")
    c.add_argument("files", nargs="*")
    c.add_argument("--subject"); c.add_argument("--body"); c.add_argument("--sender")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("serve", help="run REST API + web UI")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("connect", help="watch a mailbox and flag phishing")
    m.add_argument("provider", choices=["imap", "gmail", "outlook"])
    m.add_argument("--interval", type=int, default=60, help="seconds between mailbox checks")
    m.add_argument("--once", action="store_true", help="scan once and exit")
    m.add_argument("--dry-run", action="store_true", help="score only, do not modify mailbox")
    m.add_argument("--move", action="store_true", help="move phishing to Phishing/Junk/Spam folder")
    m.add_argument("--no-safe-tag", action="store_true", help="only tag phishing, leave legitimate untouched")
    m.set_defaults(func=cmd_connect)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
