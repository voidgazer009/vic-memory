import json
import unittest

from vic_memory import Memory
from vic_memory import dynamics as dyn
from vic_memory.mcp_server import handle

DAY = 86400.0
T0 = 1_780_000_000.0


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.m = Memory(":memory:")

    def test_remember_and_recall_en(self):
        self.m.remember("Vic prefers sea freight over air freight for heavy cargo", "semantic", at=T0)
        self.m.remember("Lunch was noodles today", at=T0)
        r = self.m.recall("which freight mode does Vic prefer", at=T0 + 60)
        self.assertTrue(r.confident)
        self.assertIn("sea freight", r.items[0].content)

    def test_recall_chinese(self):
        self.m.remember("Vic 的主营产品是 LED 工矿灯，主要出口德国和波兰", "semantic", at=T0)
        self.m.remember("今天天气不错", at=T0)
        r = self.m.recall("主营产品出口哪里", at=T0 + 60)
        self.assertIn("德国", r.items[0].content)

    def test_bitemporal_update_keeps_history(self):
        self.m.remember("Vic lives in Shenzhen", "semantic", key="user.city", at=T0)
        self.m.remember("Vic lives in Los Angeles", "semantic", key="user.city", at=T0 + 30 * DAY)
        r = self.m.recall("where does Vic live", at=T0 + 31 * DAY)
        self.assertEqual(r.items[0].content, "Vic lives in Los Angeles")
        self.assertNotIn("Shenzhen", " ".join(i.content for i in r.items))
        past = self.m.recall("where does Vic live", as_of=T0 + 10 * DAY, at=T0 + 31 * DAY)
        self.assertEqual(past.items[0].content, "Vic lives in Shenzhen")
        self.assertEqual(len(self.m.history("user.city")), 2)

    def test_duplicate_is_strengthened_not_duplicated(self):
        a = self.m.remember("Customer Hans wants CIF Hamburg quotes", at=T0)
        b = self.m.remember("Customer Hans wants CIF Hamburg quotes", at=T0 + 10)
        self.assertEqual(a.id, b.id)
        self.assertEqual(self.m.get(a.id).access_count, 2)

    def test_abstention(self):
        self.m.remember("Vic sells LED lights", "semantic", at=T0)
        r = self.m.recall("what is the capital of Peru", at=T0 + 1)
        self.assertFalse(r.confident)

    def test_forgetting_curve_and_spacing(self):
        fresh = dyn.retention([T0], T0 + 3600, 0.5)
        old = dyn.retention([T0], T0 + 60 * DAY, 0.5)
        rehearsed = dyn.retention([T0, T0 + 10 * DAY, T0 + 30 * DAY], T0 + 60 * DAY, 0.5)
        important = dyn.retention([T0], T0 + 60 * DAY, 0.95)
        self.assertGreater(fresh, old)
        self.assertGreater(rehearsed, old)
        self.assertGreater(important, old)

    def test_working_memory_gate(self):
        m = Memory(":memory:", working_capacity=3)
        m.observe("ok", at=T0)
        m.observe("Remember: my biggest client is Hans from Hamburg, always quote CIF", at=T0 + 1)
        m.observe("hmm", at=T0 + 2)
        m.observe("sure", at=T0 + 3)
        out = m.observe("thanks", at=T0 + 4)
        st = m.stats()
        self.assertEqual(sum(st.get("working", {}).values()), 3)
        r = m.recall("biggest client", at=T0 + 5)
        self.assertTrue(any("Hans" in i.content and i.layer == "long" for i in r.items))
        self.assertNotIn("ok", [i.content for i in m._items("layer='long'")])

    def test_sleep_decays_and_abstracts(self):
        for d in range(4):
            self.m.remember(f"Shipment to Hamburg delayed at port customs inspection week {d}", salience=0.5, at=T0 + d * 3600)
        self.m.remember("random note xyz", salience=0.1, at=T0)
        rep = self.m.sleep(at=T0 + 400 * DAY)
        self.assertGreaterEqual(rep["archived"], 1)
        rep2 = Memory(":memory:")
        for d in range(4):
            rep2.remember(f"Shipment to Hamburg delayed at port customs inspection week {d}", salience=0.6, at=T0 + d * 60)
        out = rep2.sleep(at=T0 + 3600)
        self.assertGreaterEqual(out["abstracted"] + out["merged"], 1)

    def test_prospective(self):
        self.m.remember("Send Hans the revised quote", "prospective", trigger={"keywords": ["Hans"]}, at=T0)
        self.m.remember("Renew ISF filing", "prospective", trigger={"at": T0 + DAY}, at=T0)
        self.assertEqual([i.content for i in self.m.due("talking about Hans", at=T0 + 1)], ["Send Hans the revised quote"])
        self.assertEqual(self.m.due("", at=T0 + 2), [])
        self.assertEqual([i.content for i in self.m.due("", at=T0 + DAY + 1)], ["Renew ISF filing"])

    def test_context_block(self):
        self.m.remember("User name is Vic, writes in Chinese", "semantic", pinned=True, at=T0)
        self.m.remember("Vic's brand is Vic, four business lines", "semantic", at=T0)
        ctx = self.m.context("brand", budget_tokens=200)
        self.assertIn("Core memory", ctx)
        self.assertIn("brand", ctx)

    def test_forget_hard(self):
        it = self.m.remember("secret phone 123", at=T0)
        self.assertEqual(self.m.forget(it.id, hard=True), 1)
        self.assertIsNone(self.m.get(it.id))

    def test_export(self):
        self.m.remember("A fact", "semantic", key="k", at=T0)
        self.m.remember("B fact", "semantic", key="k", at=T0 + 1)
        md = self.m.export_markdown()
        self.assertIn("superseded", md)

    def test_mcp_protocol(self):
        init = handle(self.m, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "vic-memory")
        tools = handle(self.m, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertIn("memory_recall", [t["name"] for t in tools])
        handle(self.m, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_remember", "arguments": {"content": "Vic likes purple glassmorphism design", "kind": "semantic"}}})
        r = handle(self.m, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "memory_recall", "arguments": {"query": "design style Vic likes"}}})
        payload = json.loads(r["result"]["content"][0]["text"])
        self.assertTrue(payload["confident"])
        self.assertIsNone(handle(self.m, {"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertIn("error", handle(self.m, {"jsonrpc": "2.0", "id": 5, "method": "nope"}))


if __name__ == "__main__":
    unittest.main()
