"""No API key needed. python examples/quickstart.py"""
from vic_memory import Memory

m = Memory(":memory:", scope="demo")

m.remember("我叫 Vic，主营 LED 工矿灯出口", "semantic", pinned=True)
m.remember("Vic's warehouse is in Shenzhen", "semantic", key="warehouse.city")
m.remember("Vic's warehouse moved to Ningbo", "semantic", key="warehouse.city")   # supersedes, keeps history
m.remember("Always quote CIF Hamburg for German buyers", "procedural")
m.remember("Send Hans the revised quote", "prospective", trigger={"keywords": ["Hans"]})

for turn in ["hi", "ok", "Remember: Hans hates long emails, keep it under 5 lines", "thanks"]:
    m.observe(turn)        # only the salient turn will survive to long-term memory

print(m.recall("where is the warehouse").text())          # -> Ningbo, not Shenzhen
print([i.content for i in m.history("warehouse.city")])   # -> full timeline
print(m.recall("what is Vic's blood type").confident)     # -> False: knows it doesn't know
print(m.context("writing an email to Hans"))              # -> ready-to-inject prompt block
print(m.sleep())
print(m.export_markdown())
