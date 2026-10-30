"""Command-line interface for the capsule prototype."""

from __future__ import annotations

import argparse
import json
import sys

from .capsule import CapsuleManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scm", description="Capsule Memory prototype")
    parser.add_argument("--store", default=".capsules/store.json", help="Path to capsule JSON store")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a capsule")
    create.add_argument("capsule_id")
    create.add_argument("--label")

    update = sub.add_parser("update", help="Update a capsule from text")
    update.add_argument("capsule_id")
    update.add_argument("text", nargs="?")
    update.add_argument("--scope", default="global")

    inspect = sub.add_parser("inspect", help="Inspect a capsule")
    inspect.add_argument("capsule_id")

    context = sub.add_parser("context", help="Render compact context payload")
    context.add_argument("capsule_id")
    context.add_argument("--max-facts", type=int, default=24)

    query = sub.add_parser("query", help="Query an exact fact")
    query.add_argument("capsule_id")
    query.add_argument("key")
    query.add_argument("--scope")

    similar = sub.add_parser("similar", help="Find capsules with similar vector sketches")
    similar.add_argument("text")
    similar.add_argument("--limit", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = CapsuleManager(args.store)

    if args.command == "create":
        capsule = manager.create(args.capsule_id, args.label)
        print(json.dumps(capsule.inspect(), indent=2))
        return 0

    if args.command == "update":
        text = args.text if args.text is not None else sys.stdin.read()
        capsule = manager.update(args.capsule_id, text, scope=args.scope)
        print(json.dumps(capsule.inspect(), indent=2))
        return 0

    if args.command == "inspect":
        print(json.dumps(manager.get(args.capsule_id).inspect(), indent=2))
        return 0

    if args.command == "context":
        print(manager.get(args.capsule_id).render_context(max_facts=args.max_facts))
        return 0

    if args.command == "query":
        result = manager.get(args.capsule_id).query_fact(args.key, scope=args.scope)
        print(json.dumps(result.__dict__, indent=2))
        return 0

    if args.command == "similar":
        print(json.dumps(manager.similar(args.text, limit=args.limit), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
