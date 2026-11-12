"""VicMemBench-lite: a reproducible, LLM-free retrieval benchmark.

It measures what a memory layer *retrieves* (not what an LLM generates), on four
abilities that separate memory from plain search (cf. LongMemEval):

  1. long-range recall   — a fact stated 1,000+ turns ago, buried in noise
  2. knowledge update    — the latest value of a fact that changed several times
  3. temporal (as-of)    — what was true at an earlier date
  4. abstention          — knowing that you don't know

Baselines:
  * window-N  : keep the last N turns in context (what a plain LLM chat does)
  * append-RAG: store every turn, BM25 top-k, no updates / no forgetting

Run:  python bench/run.py            (deterministic, seed=7, ~2 s)
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vic_memory import Memory                     # noqa: E402
from vic_memory.text import BM25, tokenize, estimate_tokens  # noqa: E402

DAY = 86400.0
T0 = 1_780_000_000.0
rng = random.Random(7)

NOISE = ["ok thanks", "sounds good", "let me check", "haha nice", "what time is it there",
         "can you rephrase that", "sure, go ahead", "I'll think about it", "good morning", "busy day today",
         "好的收到", "稍等我看一下", "哈哈不错", "今天好忙", "明白了"]
CITIES = ["Shenzhen", "Ningbo", "Hamburg", "Rotterdam", "Los Angeles", "Dubai"]
SUPPLIERS = ["Foshan Bright", "Zhongshan Lumen", "Xiamen Optic", "Hangzhou Ray"]

stable = [("Vic's main product is LED high bay lights rated 100-240W", "what is Vic's main product", "LED high bay"),
          ("Vic's biggest customer is Hans Mueller of Mueller Licht GmbH", "who is Vic's biggest customer", "Hans Mueller"),
          ("Vic is allergic to peanuts", "what food allergy does Vic have", "peanuts"),
          ("Vic's brand name is Vic and the slogan is build with agents", "what is the slogan of Vic's brand", "build with agents"),
          ("Vic 的默认贸易术语是 CIF 汉堡", "Vic 默认用什么贸易术语", "CIF"),
          ("The Poland distributor pays by 30% TT deposit and 70% against BL copy", "payment terms of the Poland distributor", "30% TT")]
unknown = ["what is Vic's blood type", "which university did Vic's sister attend", "what is the name of Vic's cat",
           "Vic 的车牌号是多少", "what is the password of the office wifi"]


def build_stream():
    turns, t = [], T0
    plan = {200: stable[0], 400: stable[1], 600: stable[2], 800: stable[3], 1000: stable[4], 1200: stable[5]}
    city_changes = {300: 0, 900: 2, 1500: 4}
    sup_changes = {500: 0, 1100: 1, 1700: 3}
    city_truth, sup_truth = [], []
    for i in range(2000):
        t += 600  # a turn every 10 minutes
        if i in plan:
            turns.append(("fact", plan[i][0], None, t))
        elif i in city_changes:
            c = CITIES[city_changes[i]]
            turns.append(("fact", f"Vic's warehouse is now located in {c}", "warehouse.city", t)); city_truth.append((t, c))
        elif i in sup_changes:
            s = SUPPLIERS[sup_changes[i]]
            turns.append(("fact", f"Vic's current main LED supplier is {s}", "supplier.main", t)); sup_truth.append((t, s))
        else:
            n = rng.choice(NOISE)
            if rng.random() < 0.15:  # distractors that share words with the facts
                n = rng.choice(["the warehouse door is squeaky", "LED screens are getting cheaper",
                                "a supplier called but I missed it", "customer service was slow today"])
            turns.append(("noise", n, None, t))
    return turns, city_truth, sup_truth, t


def truth_at(hist, at):
    v = None
    for t, x in hist:
        if t <= at:
            v = x
    return v


def main():
    turns, city_truth, sup_truth, t_end = build_stream()
    m = Memory(":memory:")
    for kind, text, key, t in turns:
        if kind == "fact":
            m.remember(text, "semantic", key=key, at=t)
        else:
            m.observe(text, at=t)
    m.sleep(at=t_end)
    rag_docs = [x[1] for x in turns]
    rag = BM25([tokenize(d) for d in rag_docs])
    window = [x[1] for x in turns[-40:]]

    def rag_top(q, k=3):
        sc = rag.scores(tokenize(q))
        return [rag_docs[i] for i in sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:k] if sc[i] > 0]

    res = {s: {"recall": 0, "update": 0, "temporal": 0, "abstain": 0} for s in ("window-40", "append-RAG", "vic-memory")}
    # 1 long-range recall
    for fact, q, ans in stable:
        res["window-40"]["recall"] += any(ans in w for w in window)
        res["append-RAG"]["recall"] += ans in (rag_top(q, 1) or [""])[0]
        r = m.recall(q, 1, at=t_end)
        res["vic-memory"]["recall"] += bool(r.items) and ans in r.items[0].content
    # 2 knowledge update
    upd = [("where is Vic's warehouse located now", city_truth[-1][1]), ("who is Vic's current main LED supplier", sup_truth[-1][1])]
    for q, ans in upd:
        res["window-40"]["update"] += any(ans in w for w in window)
        res["append-RAG"]["update"] += ans in (rag_top(q, 1) or [""])[0]
        r = m.recall(q, 1, at=t_end)
        res["vic-memory"]["update"] += bool(r.items) and ans in r.items[0].content
    # 3 temporal as-of
    probes = [(turns[0][3] + 600 * 700, "where was Vic's warehouse located", city_truth),
              (turns[0][3] + 600 * 1300, "who was Vic's main LED supplier", sup_truth),
              (turns[0][3] + 600 * 1000, "where was Vic's warehouse located", city_truth)]
    for at, q, hist in probes:
        ans = truth_at(hist, at)
        res["window-40"]["temporal"] += 0
        res["append-RAG"]["temporal"] += ans in (rag_top(q, 1) or [""])[0]
        r = m.recall(q, 1, as_of=at, at=t_end)
        res["vic-memory"]["temporal"] += bool(r.items) and ans in r.items[0].content
    # 4 abstention (correct = system signals "unknown")
    for q in unknown:
        res["window-40"]["abstain"] += 0  # a raw window gives no signal; the LLM may hallucinate
        res["append-RAG"]["abstain"] += not rag_top(q, 1)
        res["vic-memory"]["abstain"] += not m.recall(q, 1, at=t_end, reinforce=False).confident

    tot = {"recall": len(stable), "update": len(upd), "temporal": len(probes), "abstain": len(unknown)}
    ctx_tokens = {"window-40": sum(estimate_tokens(w) for w in window),
                  "append-RAG": sum(estimate_tokens(d) for d in rag_top("Vic product", 5)),
                  "vic-memory": estimate_tokens(m.context("Vic product", 800))}
    print(f"VicMemBench-lite · {len(turns)} turns over {(t_end - T0) / DAY:.1f} days · seed 7\n")
    print(f"| System | Long-range recall | Knowledge update | Temporal as-of | Abstention | Overall | Context tokens |")
    print(f"|---|---|---|---|---|---|---|")
    for s, r in res.items():
        overall = sum(r.values()) / sum(tot.values())
        cells = " | ".join(f"{r[k]}/{tot[k]}" for k in tot)
        print(f"| {s} | {cells} | {overall:.0%} | {ctx_tokens[s]} |")
    print(f"\nvic-memory store after sleep: {m.stats()}")


if __name__ == "__main__":
    main()
