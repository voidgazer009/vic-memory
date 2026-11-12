"""Model Context Protocol server (stdio, JSON-RPC 2.0) — zero dependencies.

Plug Vic Memory into Claude Desktop/Code, Cursor, Codex, Cline, Windsurf, or any MCP client:

    {"mcpServers": {"vic-memory": {"command": "python", "args": ["-m", "vic_memory.mcp_server"],
                                   "env": {"VIC_MEMORY_DB": "~/.vic_memory.db"}}}}
"""
from __future__ import annotations

import json
import os
import sys

from . import __version__
from .core import Memory, KINDS
from .llm import from_env

PROTOCOL = "2025-06-18"

TOOLS = [
    {"name": "memory_remember", "description": "Store a durable memory. Use kind=semantic with a key (e.g. 'user.city') for facts that can change; newer values supersede older ones but history is kept. kind=procedural for how-to/lessons, prospective for future intentions (with trigger).",
     "inputSchema": {"type": "object", "properties": {
         "content": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS), "default": "episodic"},
         "key": {"type": "string"}, "salience": {"type": "number", "minimum": 0, "maximum": 1},
         "pinned": {"type": "boolean", "description": "Core memory always shown in context"},
         "tags": {"type": "array", "items": {"type": "string"}},
         "trigger": {"type": "object", "description": "For prospective: {at: unix_ts} and/or {keywords: [...]}"}},
         "required": ["content"]}},
    {"name": "memory_observe", "description": "Feed a raw dialogue turn or observation. It passes a salience gate into working memory and is promoted to long-term memory only if it matters.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "memory_recall", "description": "Retrieve memories relevant to a cue. Returns confident=false when nothing reliable is known (then say you don't know). as_of (unix ts) answers what was true at that time.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5},
                     "kinds": {"type": "array", "items": {"type": "string", "enum": list(KINDS)}},
                     "as_of": {"type": "number"}}, "required": ["query"]}},
    {"name": "memory_context", "description": "Get a ready-to-use memory block for the current turn: core facts, due reminders, relevant memories, recent working memory.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "budget_tokens": {"type": "integer", "default": 800}}}},
    {"name": "memory_history", "description": "Every value a semantic key has had over time.",
     "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
    {"name": "memory_forget", "description": "Forget a memory by id or by query. Soft (archive) unless hard=true.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "query": {"type": "string"}, "hard": {"type": "boolean", "default": False}}}},
    {"name": "memory_sleep", "description": "Run offline consolidation: decay weak memories, merge duplicates, abstract episodes into knowledge. Call at session end or periodically.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_stats", "description": "Counts per layer and kind.", "inputSchema": {"type": "object", "properties": {}}},
]


def _mem() -> Memory:
    path = os.path.expanduser(os.environ.get("VIC_MEMORY_DB", "~/.vic_memory.db"))
    emb, summ = from_env()
    return Memory(path, scope=os.environ.get("VIC_MEMORY_SCOPE", "default"), embedder=emb, summarizer=summ)


def call_tool(m: Memory, name: str, a: dict):
    if name == "memory_remember":
        it = m.remember(a["content"], a.get("kind", "episodic"), key=a.get("key"), salience=a.get("salience"),
                        pinned=bool(a.get("pinned", False)), tags=a.get("tags", []), trigger=a.get("trigger"),
                        source="mcp")
        return {"id": it.id, "kind": it.kind, "key": it.key, "salience": round(it.salience, 3)}
    if name == "memory_observe":
        return m.observe(a["text"], source="mcp")
    if name == "memory_recall":
        r = m.recall(a["query"], k=int(a.get("k", 5)), kinds=a.get("kinds"), as_of=a.get("as_of"))
        return {"confident": r.confident, "memories": [
            {"id": i.id, "kind": i.kind, "key": i.key, "content": i.content, "score": round(i.score, 3),
             "strength": round(i.strength, 3)} for i in r.items]}
    if name == "memory_context":
        return m.context(a.get("query", ""), int(a.get("budget_tokens", 800)))
    if name == "memory_history":
        return [{"content": i.content, "valid_from": i.valid_from, "valid_to": i.valid_to} for i in m.history(a["key"])]
    if name == "memory_forget":
        return {"forgotten": m.forget(a.get("id"), query=a.get("query"), hard=bool(a.get("hard", False)))}
    if name == "memory_sleep":
        return m.sleep()
    if name == "memory_stats":
        return m.stats()
    raise KeyError(name)


def handle(m: Memory, msg: dict):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:  # notification
        return None
    try:
        if method == "initialize":
            res = {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "vic-memory", "version": __version__},
                   "instructions": "Call memory_context at the start of a turn, memory_observe on each user turn, "
                                   "memory_remember for durable facts, memory_sleep at session end."}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": TOOLS}
        elif method == "tools/call":
            p = msg.get("params", {})
            try:
                out = call_tool(m, p.get("name"), p.get("arguments") or {})
                text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                res = {"content": [{"type": "text", "text": text}], "isError": False}
            except KeyError as e:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool or missing argument: {e}"}}
            except Exception as e:  # tool-level error, reported to the model
                res = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": res}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}


def main() -> None:
    m = _mem()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}) + "\n")
            sys.stdout.flush()
            continue
        resp = handle(m, msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
