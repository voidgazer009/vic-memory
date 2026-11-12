"""Tokenisation and lexical scoring.

Zero-dependency, bilingual (Chinese + English) tokeniser and BM25.
CJK text is indexed as unigrams + bigrams, which works well without a
segmenter; Latin text as lowercase words.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_WORD = re.compile(r"[a-z0-9][a-z0-9_\-\.']*", re.I)
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "it", "this", "that", "with", "as", "at", "by", "i",
    "you", "he", "she", "we", "they", "my", "your", "me", "do", "does", "did",
    "what", "which", "who", "when", "where", "how", "s", "t",
    "的", "了", "是", "在", "和", "也", "就", "都", "而", "及", "与", "着", "吗", "呢", "吧",
}


# CJK function words / question words: the run is split on these before
# n-gramming, so "喜欢什么设计" -> "喜欢" + "设计" (no junk bigrams like "欢什").
_CJK_STOP = re.compile(
    "为什么|是不是|有没有|什么|怎么|怎样|哪里|哪儿|哪个|多少|如何|是否|请问|一下|现在|"
    "[的了是在和也就都而及与着吗呢吧啊呀么哪谁几把被给让这那我你他她它们请]"
)


def stem(w: str) -> str:
    """Tiny suffix stripper so 'prefers'/'preferred'/'preferring' meet 'prefer'."""
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    """Return search tokens for *text* (Latin words + CJK uni/bi-grams)."""
    if not text:
        return []
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        w = w.strip(".'-")
        if w and w not in STOPWORDS:
            out.append(stem(w))
    for run in _CJK.findall(text):
        for piece in _CJK_STOP.split(run):
            if not piece:
                continue
            out.extend(piece)                                        # unigrams
            out.extend(piece[i:i + 2] for i in range(len(piece) - 1))  # bigrams
    return out


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def estimate_tokens(text: str) -> int:
    """Rough LLM-token estimate: ~1 token per CJK char, ~4 chars per token otherwise."""
    cjk = sum(len(r) for r in _CJK.findall(text))
    return cjk + max(0, len(text) - cjk) // 4 + 1


class BM25:
    """Okapi BM25 over an in-memory corpus of pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: Sequence[str], i: int) -> float:
        d, dl = self.docs[i], self.lens[i]
        if not dl:
            return 0.0
        s = 0.0
        for t in set(query):
            tf = d.get(t)
            if not tf:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def scores(self, query: Sequence[str]) -> list[float]:
        return [self.score(query, i) for i in range(len(self.docs))]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
