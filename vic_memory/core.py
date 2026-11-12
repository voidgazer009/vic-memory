"""Vic Memory — a brain-inspired, model-agnostic memory for any agent.

Layers (mirroring the human memory system):

    observe() ──► SENSORY GATE ──► WORKING MEMORY (K slots) ──► LONG-TERM
                  salience+novelty   overflow is promoted       episodic / semantic /
                                     only if salient            procedural / prospective
                                                                   │
                         recall()  ◄── hybrid cue retrieval ◄──────┤  (lexical + vector +
                                       + spreading activation      │   activation + salience)
                                       + reconsolidation           │
                         sleep()   ──► decay / dedupe / abstract / resolve (offline)

Everything is stored in one SQLite file. No third-party dependencies.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional, Sequence

from . import dynamics as dyn
from .text import BM25, tokenize, jaccard, cosine, estimate_tokens

KINDS = ("episodic", "semantic", "procedural", "prospective")
Embedder = Callable[[str], Sequence[float]]
Summarizer = Callable[[list[str]], str]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'default',
    layer TEXT NOT NULL DEFAULT 'long',          -- working | long | archive
    kind TEXT NOT NULL DEFAULT 'episodic',
    content TEXT NOT NULL,
    key TEXT,                                     -- semantic slot, e.g. user.city
    salience REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.8,
    source TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    event_time REAL,                              -- when it happened in the world
    valid_from REAL NOT NULL,
    valid_to REAL,                                -- NULL = still true (bi-temporal)
    superseded_by TEXT,
    accesses TEXT NOT NULL DEFAULT '[]',          -- recent access timestamps
    access_count INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    consolidated INTEGER NOT NULL DEFAULT 0,
    trigger TEXT,                                 -- prospective memory trigger JSON
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS ix_scope_layer ON memories(scope, layer);
CREATE INDEX IF NOT EXISTS ix_key ON memories(scope, key);
CREATE TABLE IF NOT EXISTS links (
    a TEXT NOT NULL, b TEXT NOT NULL, weight REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY (a, b)
);
"""

MAX_TRACES = 32


@dataclass
class Item:
    id: str
    content: str
    kind: str = "episodic"
    scope: str = "default"
    layer: str = "long"
    key: Optional[str] = None
    salience: float = 0.5
    confidence: float = 0.8
    source: Optional[str] = None
    tags: list = field(default_factory=list)
    created_at: float = 0.0
    event_time: Optional[float] = None
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    superseded_by: Optional[str] = None
    accesses: list = field(default_factory=list)
    access_count: int = 0
    pinned: bool = False
    consolidated: bool = False
    trigger: Optional[dict] = None
    embedding: Optional[list] = None
    score: float = 0.0
    strength: float = 0.0

    def to_dict(self, slim: bool = True) -> dict:
        d = asdict(self)
        if slim:
            for k in ("embedding", "accesses"):
                d.pop(k, None)
        return d


@dataclass
class Recall:
    items: list
    confident: bool          # metamemory: do we actually know?
    query: str

    def text(self) -> str:
        return "\n".join(f"- {i.content}" for i in self.items)


class Memory:
    """The whole memory system. Thread-safe; one instance per SQLite file.

    Parameters
    ----------
    path:        SQLite file (":memory:" for ephemeral).
    scope:       namespace (user id, agent id, project...).
    embedder:    optional ``f(text) -> vector`` for semantic similarity.
    summarizer:  optional ``f(list_of_texts) -> str`` used by sleep() to
                 abstract episodes into semantic knowledge (any LLM).
    working_capacity: slots in working memory (Cowan: ~4±1; default 7).
    """

    def __init__(self, path: str = "vic_memory.db", scope: str = "default", *,
                 embedder: Optional[Embedder] = None,
                 summarizer: Optional[Summarizer] = None,
                 working_capacity: int = 7,
                 gate_threshold: float = 0.45,
                 forget_threshold: float = 0.08,
                 abstain_threshold: float = 0.35):
        self.path, self.scope = path, scope
        self.embedder, self.summarizer = embedder, summarizer
        self.working_capacity = working_capacity
        self.gate_threshold = gate_threshold
        self.forget_threshold = forget_threshold
        self.abstain_threshold = abstain_threshold
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # ------------------------------------------------------------------ utils
    def _row(self, r: sqlite3.Row) -> Item:
        d = dict(r)
        d["tags"] = json.loads(d["tags"] or "[]")
        d["accesses"] = json.loads(d["accesses"] or "[]")
        d["pinned"] = bool(d["pinned"])
        d["consolidated"] = bool(d["consolidated"])
        d["trigger"] = json.loads(d["trigger"]) if d["trigger"] else None
        d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
        return Item(**d)

    def _items(self, where: str = "1=1", args: Iterable = ()) -> list[Item]:
        q = f"SELECT * FROM memories WHERE scope=? AND {where}"
        return [self._row(r) for r in self.db.execute(q, (self.scope, *args))]

    def get(self, mid: str) -> Optional[Item]:
        r = self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        return self._row(r) if r else None

    def _insert(self, it: Item) -> None:
        d = it.to_dict(slim=False)
        d.pop("score"), d.pop("strength")
        d["tags"] = json.dumps(d["tags"], ensure_ascii=False)
        d["accesses"] = json.dumps(d["accesses"])
        d["trigger"] = json.dumps(d["trigger"], ensure_ascii=False) if d["trigger"] else None
        d["embedding"] = json.dumps(d["embedding"]) if d["embedding"] else None
        d["pinned"], d["consolidated"] = int(d["pinned"]), int(d["consolidated"])
        cols = ",".join(d)
        self.db.execute(f"INSERT INTO memories ({cols}) VALUES ({','.join('?' * len(d))})", tuple(d.values()))

    def _touch(self, it: Item, at: float) -> None:
        """Reconsolidation: each retrieval adds an access trace (strengthens)."""
        acc = (it.accesses + [at])[-MAX_TRACES:]
        self.db.execute("UPDATE memories SET accesses=?, access_count=access_count+1 WHERE id=?",
                        (json.dumps(acc), it.id))

    def _link(self, a: str, b: str, w: float) -> None:
        for x, y in ((a, b), (b, a)):
            self.db.execute("INSERT INTO links(a,b,weight) VALUES(?,?,?) "
                            "ON CONFLICT(a,b) DO UPDATE SET weight=MIN(1.0, weight+excluded.weight*0.5)",
                            (x, y, w))

    def neighbors(self, mid: str) -> list[tuple[str, float]]:
        return [(r["b"], r["weight"]) for r in self.db.execute("SELECT b, weight FROM links WHERE a=?", (mid,))]

    # ---------------------------------------------------------------- encoding
    def remember(self, content: str, kind: str = "episodic", *, key: Optional[str] = None,
                 salience: Optional[float] = None, confidence: float = 0.8,
                 source: Optional[str] = None, tags: Sequence[str] = (),
                 event_time: Optional[float] = None, pinned: bool = False,
                 trigger: Optional[dict] = None, at: Optional[float] = None) -> Item:
        """Encode directly into long-term memory (explicit memory, bypasses the gate).

        * ``key`` makes it a semantic *slot*: a newer value for the same key
          invalidates the old one (``valid_to`` set, ``superseded_by`` linked)
          instead of deleting it — so the past stays answerable.
        * Near-duplicates are merged and strengthened rather than re-stored.
        * New memories are auto-linked to related ones (associative network).
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        content = (content or "").strip()
        if not content:
            raise ValueError("empty memory")
        t = at if at is not None else dyn.now()
        sal = dyn.estimate_salience(content) if salience is None else float(salience)
        with self._lock:
            toks = tokenize(content)
            live = self._items("layer!='archive' AND valid_to IS NULL")
            # 1) duplicate -> strengthen (repetition effect)
            for old in live:
                if old.kind == kind and old.key == key and jaccard(toks, tokenize(old.content)) >= 0.9:
                    self._touch(old, t)
                    self.db.execute("UPDATE memories SET salience=MAX(salience,?) WHERE id=?", (sal, old.id))
                    self.db.commit()
                    return self.get(old.id)
            it = Item(id=uuid.uuid4().hex[:12], content=content, kind=kind, scope=self.scope,
                      key=key, salience=sal, confidence=confidence, source=source, tags=list(tags),
                      created_at=t, event_time=event_time, valid_from=event_time or t,
                      accesses=[t], access_count=1, pinned=pinned, trigger=trigger)
            if self.embedder:
                it.embedding = [float(x) for x in self.embedder(content)]
            self._insert(it)
            # 2) slot conflict -> invalidate older belief (bi-temporal update)
            if key:
                for old in live:
                    if old.key == key and old.id != it.id:
                        self.db.execute("UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                                        (it.valid_from, it.id, old.id))
                        self._link(it.id, old.id, 0.6)
            # 3) associative linking (Zettelkasten / hippocampal index)
            if live:
                bm = BM25([tokenize(o.content) for o in live])
                sc = bm.scores(toks)
                top = sorted(range(len(live)), key=lambda i: sc[i], reverse=True)[:5]
                for i in top:
                    if sc[i] <= 0:
                        break
                    sim = jaccard(toks, tokenize(live[i].content))
                    if sim >= 0.2:
                        self._link(it.id, live[i].id, round(sim, 3))
            self.db.commit()
            return it

    def observe(self, text: str, source: str = "dialogue", at: Optional[float] = None) -> dict:
        """Feed raw experience (each dialogue turn, tool result...).

        Goes through the sensory gate into working memory. When working memory
        overflows, the oldest item is promoted to long-term episodic memory if
        it is salient enough, otherwise it is forgotten — like real short-term memory.
        """
        t = at if at is not None else dyn.now()
        sal = dyn.estimate_salience(text)
        with self._lock:
            novel = all(jaccard(tokenize(text), tokenize(o.content)) < 0.9
                        for o in self._items("layer='working'"))
            if not novel:
                return {"stored": False, "reason": "duplicate", "salience": sal}
            it = Item(id=uuid.uuid4().hex[:12], content=text.strip(), scope=self.scope, layer="working",
                      salience=sal, source=source, created_at=t, valid_from=t, accesses=[t], access_count=1)
            self._insert(it)
            promoted, dropped = [], []
            wm = sorted(self._items("layer='working'"), key=lambda x: x.created_at)
            while len(wm) > self.working_capacity:
                old = wm.pop(0)
                self.db.execute("DELETE FROM memories WHERE id=?", (old.id,))
                if old.salience >= self.gate_threshold:
                    self.db.commit()
                    self.remember(old.content, "episodic", salience=old.salience, source=old.source,
                                  event_time=old.created_at, at=t)
                    promoted.append(old.content)
                else:
                    dropped.append(old.content)
            self.db.commit()
            return {"stored": True, "salience": round(sal, 3), "promoted": promoted, "dropped": dropped}

    def flush_working(self) -> int:
        """End of session: promote salient working-memory items, clear the rest."""
        with self._lock:
            n = 0
            for it in self._items("layer='working'"):
                self.db.execute("DELETE FROM memories WHERE id=?", (it.id,))
                if it.salience >= self.gate_threshold:
                    self.db.commit()
                    self.remember(it.content, "episodic", salience=it.salience, source=it.source,
                                  event_time=it.created_at)
                    n += 1
            self.db.commit()
            return n

    # --------------------------------------------------------------- retrieval
    def recall(self, query: str, k: int = 5, *, kinds: Optional[Sequence[str]] = None,
               as_of: Optional[float] = None, include_archive: bool = False,
               spread: bool = True, reinforce: bool = True, at: Optional[float] = None) -> Recall:
        """Cue-based retrieval.

        score = 0.45·lexical + 0.25·semantic(vector) + 0.15·activation + 0.10·salience
                + 0.05·confidence, then spreading activation along associative
        links. ``as_of`` answers "what was true at time T" (bi-temporal).
        Retrieved memories are strengthened (testing effect / reconsolidation).
        """
        t = at if at is not None else dyn.now()
        with self._lock:
            layers = "('long','working','archive')" if include_archive else "('long','working')"
            cands = self._items(f"layer IN {layers}")
            if kinds:
                cands = [c for c in cands if c.kind in kinds]
            if as_of is None:
                cands = [c for c in cands if c.valid_to is None]
            else:
                cands = [c for c in cands if c.valid_from <= as_of and (c.valid_to is None or c.valid_to > as_of)]
            if not cands:
                return Recall([], False, query)
            qt = tokenize(query)
            bm = BM25([tokenize(c.content) for c in cands])
            lex = bm.scores(qt)
            mx = max(lex) or 1.0
            qv = list(self.embedder(query)) if self.embedder else None
            by_id = {}
            for c, l in zip(cands, lex):
                sem = cosine(qv, c.embedding) if (qv and c.embedding) else 0.0
                c.strength = dyn.retention(c.accesses, t, c.salience)
                rel = 0.45 * (l / mx) + 0.25 * max(sem, 0.0)
                c.score = rel + (0.15 * c.strength + 0.10 * c.salience + 0.05 * c.confidence) * (1 if rel > 0 else 0)
                by_id[c.id] = c
            if spread:  # spreading activation (Collins & Loftus) from the top seeds
                seeds = sorted(cands, key=lambda c: c.score, reverse=True)[:k]
                for s in seeds:
                    for nid, w in self.neighbors(s.id):
                        if nid in by_id and by_id[nid].score < s.score:
                            by_id[nid].score += 0.3 * w * s.score
            ranked = sorted((c for c in cands if c.score > 0), key=lambda c: c.score, reverse=True)[:k]
            confident = bool(ranked) and ranked[0].score >= self.abstain_threshold and (
                self._coverage(qt, ranked[0], bm, len(cands)) >= 0.3
                or (qv is not None and ranked[0].embedding is not None and cosine(qv, ranked[0].embedding) >= 0.5))
            if reinforce:
                for c in ranked:
                    self._touch(c, t)
                self.db.commit()
            return Recall(ranked, confident, query)

    @staticmethod
    def _coverage(qt: Sequence[str], top: Item, bm: BM25, n: int) -> float:
        """Metamemory: IDF-weighted share of the cue that the best memory actually covers.
        Common words (high document frequency) count little; rare cue words count a lot."""
        import math
        q = set(qt)
        if not q:
            return 0.0
        unseen = math.log(1 + (n + 0.5) / 0.5)
        have = set(tokenize(top.content))
        tot = sum(bm.idf.get(t, unseen) for t in q)
        hit = sum(bm.idf.get(t, unseen) for t in q if t in have)
        return hit / tot if tot else 0.0

    def history(self, key: str) -> list[Item]:
        """All values a semantic slot has had, oldest first."""
        return sorted(self._items("key=?", (key,)), key=lambda x: x.valid_from)

    def due(self, context: str = "", at: Optional[float] = None) -> list[Item]:
        """Prospective memory: intentions whose trigger fires now.

        trigger = {"at": unix_ts} and/or {"keywords": [...]}; fired ones are archived.
        """
        t = at if at is not None else dyn.now()
        toks = set(tokenize(context))
        fired = []
        with self._lock:
            for it in self._items("kind='prospective' AND layer='long' AND valid_to IS NULL"):
                tr = it.trigger or {}
                hit = ("at" in tr and t >= float(tr["at"])) or \
                      any(set(tokenize(k)) <= toks for k in tr.get("keywords", []) if tokenize(k))
                if hit:
                    fired.append(it)
                    self.db.execute("UPDATE memories SET layer='archive', valid_to=? WHERE id=?", (t, it.id))
            self.db.commit()
        return fired

    def context(self, query: str = "", budget_tokens: int = 800, k: int = 8) -> str:
        """Assemble a prompt block (the agent's 'working memory' for this turn):
        pinned core facts → due intentions → recalled long-term → recent working memory."""
        parts, used = [], 0

        def add(title: str, lines: list[str]):
            nonlocal used
            keep = []
            for ln in lines:
                c = estimate_tokens(ln)
                if used + c > budget_tokens:
                    break
                keep.append(ln)
                used += c
            if keep:
                parts.append(f"## {title}\n" + "\n".join(keep))

        core = self._items("pinned=1 AND valid_to IS NULL AND layer!='archive'")
        add("Core memory", [f"- {c.content}" for c in core])
        add("Reminders due", [f"- {d.content}" for d in self.due(query)])
        if query:
            core_ids = {c.id for c in core}
            r = self.recall(query, k=k)
            add("Relevant memories", [f"- [{i.kind}] {i.content}" for i in r.items
                                      if i.id not in core_ids and i.layer != "working"])
            if not r.confident:
                parts.append("_(No confident long-term memory for this query — say you don't know rather than guess.)_")
        wm = sorted(self._items("layer='working'"), key=lambda x: x.created_at)
        add("Working memory (recent)", [f"- {w.content}" for w in wm])
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- forgetting
    def forget(self, mid: Optional[str] = None, *, query: Optional[str] = None, hard: bool = False) -> int:
        """Explicit forgetting. Soft (archive) by default; ``hard=True`` deletes (GDPR)."""
        with self._lock:
            ids = [mid] if mid else []
            if query:
                ids += [i.id for i in self.recall(query, k=50, reinforce=False, spread=False).items]
            for i in ids:
                if hard:
                    self.db.execute("DELETE FROM memories WHERE id=?", (i,))
                    self.db.execute("DELETE FROM links WHERE a=? OR b=?", (i, i))
                else:
                    self.db.execute("UPDATE memories SET layer='archive' WHERE id=?", (i,))
            self.db.commit()
            return len(ids)

    # ---------------------------------------------------------------- sleep
    def sleep(self, at: Optional[float] = None, min_cluster: int = 3) -> dict:
        """Offline consolidation, like slow-wave sleep. Run it periodically.

        1. Decay   — weak, unimportant, unpinned memories move to the archive.
        2. Dedupe  — near-duplicates merge; traces are pooled (stronger memory).
        3. Abstract— clusters of linked episodes become one semantic memory
                     (uses ``summarizer`` if given, else an extractive gist).
        """
        t = at if at is not None else dyn.now()
        rep = {"archived": 0, "merged": 0, "abstracted": 0}
        with self._lock:
            live = self._items("layer='long' AND valid_to IS NULL")
            for it in live:
                if it.pinned or it.kind == "prospective":
                    continue
                if dyn.retention(it.accesses, t, it.salience) < self.forget_threshold and it.salience < 0.7:
                    self.db.execute("UPDATE memories SET layer='archive' WHERE id=?", (it.id,))
                    rep["archived"] += 1
            live = self._items("layer='long' AND valid_to IS NULL")
            gone: set = set()
            for i, a in enumerate(live):
                if a.id in gone:
                    continue
                ta = tokenize(a.content)
                for b in live[i + 1:]:
                    if b.id in gone or b.kind != a.kind or b.key != a.key:
                        continue
                    if jaccard(ta, tokenize(b.content)) >= 0.8:
                        acc = sorted(a.accesses + b.accesses)[-MAX_TRACES:]
                        self.db.execute("UPDATE memories SET accesses=?, salience=MAX(salience,?), "
                                        "access_count=access_count+? WHERE id=?",
                                        (json.dumps(acc), b.salience, b.access_count, a.id))
                        self.db.execute("UPDATE memories SET layer='archive', superseded_by=? WHERE id=?",
                                        (a.id, b.id))
                        gone.add(b.id)
                        rep["merged"] += 1
            episodes = [e for e in self._items("layer='long' AND kind='episodic' AND consolidated=0 AND valid_to IS NULL")]
            ep_ids = {e.id: e for e in episodes}
            seen: set = set()
            for e in episodes:
                if e.id in seen:
                    continue
                cluster, stack = [], [e.id]
                while stack:  # connected component over associative links
                    x = stack.pop()
                    if x in seen or x not in ep_ids:
                        continue
                    seen.add(x)
                    cluster.append(ep_ids[x])
                    stack.extend(n for n, w in self.neighbors(x) if w >= 0.2)
                if len(cluster) < min_cluster:
                    continue
                cluster.sort(key=lambda c: c.created_at)
                texts = [c.content for c in cluster]
                gist = self.summarizer(texts) if self.summarizer else self._extractive_gist(texts)
                sem = self.remember(gist, "semantic", salience=max(c.salience for c in cluster),
                                    source="sleep:consolidation", confidence=0.7, at=t)
                for c in cluster:
                    self._link(sem.id, c.id, 0.8)
                    self.db.execute("UPDATE memories SET consolidated=1 WHERE id=?", (c.id,))
                rep["abstracted"] += 1
            self.db.commit()
        return rep

    @staticmethod
    def _extractive_gist(texts: list[str], max_chars: int = 400) -> str:
        """No-LLM fallback: keep the sentences that share the most vocabulary with the cluster."""
        from collections import Counter
        bag = Counter(t for s in texts for t in set(tokenize(s)))
        ranked = sorted(texts, key=lambda s: -sum(bag[t] for t in set(tokenize(s))) / (1 + len(set(tokenize(s))) ** 0.5))
        out = ""
        for s in ranked:
            if len(out) + len(s) > max_chars:
                break
            out += ("; " if out else "") + s
        return "Pattern: " + (out or texts[0][:max_chars])

    # ---------------------------------------------------------------- export
    def stats(self) -> dict:
        rows = self.db.execute("SELECT layer, kind, COUNT(*) n FROM memories WHERE scope=? GROUP BY layer, kind",
                               (self.scope,)).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["layer"], {})[r["kind"]] = r["n"]
        out["links"] = self.db.execute("SELECT COUNT(*) FROM links").fetchone()[0] // 2
        return out

    def export_markdown(self) -> str:
        """Human-readable, diffable dump — the memory is never a black box."""
        lines = [f"# Vic Memory export — scope `{self.scope}`", ""]
        for kind in KINDS:
            items = self._items("kind=? AND layer!='working'", (kind,))
            if not items:
                continue
            lines.append(f"## {kind}")
            for i in sorted(items, key=lambda x: x.valid_from):
                flags = []
                if i.pinned: flags.append("📌")
                if i.kind == "prospective" and i.valid_to: flags.append("✅ done")
                elif i.valid_to: flags.append("~~superseded~~")
                if i.layer == "archive" and i.kind != "prospective": flags.append("(archived)")
                key = f"`{i.key}` " if i.key else ""
                lines.append(f"- {key}{i.content} {' '.join(flags)}  <sub>{i.id} · salience {i.salience:.2f}</sub>")
            lines.append("")
        return "\n".join(lines)

    def close(self) -> None:
        self.db.close()
