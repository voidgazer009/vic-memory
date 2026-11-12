"""Command line: vic-memory remember|observe|recall|context|sleep|stats|export|mcp"""
from __future__ import annotations

import argparse
import json
import os

from .core import Memory, KINDS
from .llm import from_env


def main(argv=None) -> None:
    p = argparse.ArgumentParser("vic-memory", description="Brain-inspired memory for any agent")
    p.add_argument("--db", default=os.path.expanduser(os.environ.get("VIC_MEMORY_DB", "~/.vic_memory.db")))
    p.add_argument("--scope", default=os.environ.get("VIC_MEMORY_SCOPE", "default"))
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("remember"); r.add_argument("content"); r.add_argument("--kind", choices=KINDS, default="episodic")
    r.add_argument("--key"); r.add_argument("--salience", type=float); r.add_argument("--pin", action="store_true")
    o = sub.add_parser("observe"); o.add_argument("text")
    q = sub.add_parser("recall"); q.add_argument("query"); q.add_argument("-k", type=int, default=5)
    c = sub.add_parser("context"); c.add_argument("query", nargs="?", default=""); c.add_argument("--budget", type=int, default=800)
    h = sub.add_parser("history"); h.add_argument("key")
    f = sub.add_parser("forget"); f.add_argument("--id"); f.add_argument("--query"); f.add_argument("--hard", action="store_true")
    sub.add_parser("sleep"); sub.add_parser("stats"); sub.add_parser("export"); sub.add_parser("mcp")
    a = p.parse_args(argv)
    if a.cmd == "mcp":
        from .mcp_server import main as mcp_main
        return mcp_main()
    emb, summ = from_env()
    m = Memory(a.db, a.scope, embedder=emb, summarizer=summ)
    if a.cmd == "remember":
        it = m.remember(a.content, a.kind, key=a.key, salience=a.salience, pinned=a.pin, source="cli")
        print(f"remembered {it.id} [{it.kind}] salience={it.salience:.2f}")
    elif a.cmd == "observe":
        print(json.dumps(m.observe(a.text, source="cli"), ensure_ascii=False))
    elif a.cmd == "recall":
        res = m.recall(a.query, a.k)
        print(f"confident={res.confident}")
        for i in res.items:
            print(f"{i.score:.3f}  [{i.kind}] {i.content}")
    elif a.cmd == "context":
        print(m.context(a.query, a.budget))
    elif a.cmd == "history":
        for i in m.history(a.key):
            print(("  (old) " if i.valid_to else "  (now) ") + i.content)
    elif a.cmd == "forget":
        print(m.forget(a.id, query=a.query, hard=a.hard))
    elif a.cmd == "sleep":
        print(json.dumps(m.sleep()))
    elif a.cmd == "stats":
        print(json.dumps(m.stats(), indent=2))
    elif a.cmd == "export":
        print(m.export_markdown())


if __name__ == "__main__":
    main()
