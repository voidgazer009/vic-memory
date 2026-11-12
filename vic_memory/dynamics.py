"""Memory dynamics borrowed from cognitive science.

* ACT-R base-level activation:   B = ln( sum_j  t_j ^ -d )
  Every access (encoding or recall) adds a trace t_j; recent and frequent
  memories are more active. This yields both the Ebbinghaus forgetting curve
  and the spacing effect for free.
* Importance / emotional salience slows decay (amygdala modulation):
  d = d0 * (1 - 0.5 * salience)
* Sensory gate: a cheap salience estimate decides whether a raw observation
  is worth encoding into long-term memory at all.
"""
from __future__ import annotations

import math
import re
import time

HOUR = 3600.0

# Cue phrases that signal a memory the user wants kept (EN + ZH).
_SALIENT = re.compile(
    r"\b(remember|important|always|never|prefer|favou?rite|allerg\w*|deadline|"
    r"my name|call me|i am|i'm|i live|i work|birthday|password|don't|do not|must)\b"
    r"|记住|重要|一定|总是|永远|从不|不要|别|喜欢|讨厌|偏好|过敏|截止|生日|我叫|我是|我住|我在|必须",
    re.I,
)
_EMOTION = re.compile(
    r"\b(love|hate|angry|furious|excited|thrilled|worried|afraid|sad|happy|amazing|terrible)\b"
    r"|爱|恨|生气|愤怒|激动|开心|兴奋|担心|害怕|难过|伤心|太棒|糟糕|崩溃",
    re.I,
)


def now() -> float:
    return time.time()


def base_activation(access_times: list[float], at: float, salience: float, d0: float = 0.5) -> float:
    """ACT-R base-level activation with salience-modulated decay."""
    d = d0 * (1.0 - 0.5 * max(0.0, min(1.0, salience)))
    total = 0.0
    for t in access_times:
        age_h = max((at - t) / HOUR, 1.0 / 60)  # floor at one minute
        total += age_h ** (-d)
    return math.log(total) if total > 0 else -10.0


def retention(access_times: list[float], at: float, salience: float) -> float:
    """Probability-like strength in (0,1) — how retrievable the memory is now."""
    b = base_activation(access_times, at, salience)
    x = b + 3.0 * (salience - 0.5) + 1.0
    return 1.0 / (1.0 + math.exp(-x))


def estimate_salience(text: str) -> float:
    """Heuristic importance in [0,1] used when no LLM judge is configured."""
    if not text:
        return 0.0
    s = 0.3
    s += 0.35 * min(2, len(_SALIENT.findall(text))) / 2
    s += 0.2 * min(2, len(_EMOTION.findall(text))) / 2
    if re.search(r"\d", text):
        s += 0.05  # numbers: dates, prices, quantities
    if len(text) < 6:
        s -= 0.2
    return max(0.0, min(1.0, s))
