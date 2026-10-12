"""Run with:  python -m unittest discover -s tests -v   (standard library only)"""
import copy
import unittest

from ratchet.agents import NAIVE_READING, ScriptedAnalyst
from ratchet.core import LocalMemory, PolicyDenied, PolicyEngine, PolicyRejected, SourceSpec
from ratchet.loop import ORG_POLICY, evaluate, replay, run_ratchet
from ratchet.world import (DEFINITIONS, LocalTools, aggregate, build_dataset, load_world, matches, schema_of)


class PolicyCeiling(unittest.TestCase):
    def setUp(self):
        self.pe = PolicyEngine(org={"AUTHORIZED_TOOLS": ["a", "b"], "MAX_TOOL_CALLS_PER_EXECUTION": 4})

    def test_project_cannot_widen_allowlist(self):
        with self.assertRaises(PolicyRejected):
            self.pe.set_project("AUTHORIZED_TOOLS", ["a", "c"])
        self.assertNotIn("AUTHORIZED_TOOLS", self.pe.project)

    def test_project_cannot_raise_cap(self):
        with self.assertRaises(PolicyRejected):
            self.pe.set_project("MAX_TOOL_CALLS_PER_EXECUTION", 5)

    def test_effective_is_intersection_and_minimum(self):
        self.pe.set_project("AUTHORIZED_TOOLS", ["a"])
        self.pe.set_project("MAX_TOOL_CALLS_PER_EXECUTION", 2)
        self.assertEqual(self.pe.effective(), {"AUTHORIZED_TOOLS": ["a"], "MAX_TOOL_CALLS_PER_EXECUTION": 2})

    def test_no_policy_means_unrestricted(self):
        PolicyEngine().check_tool_call("anything", 1000)

    def test_denials(self):
        self.pe.set_project("MAX_TOOL_CALLS_PER_EXECUTION", 1)
        with self.assertRaises(PolicyDenied) as cm:
            self.pe.check_tool_call("zzz", 0)
        self.assertTrue(str(cm.exception).startswith("Policy denied: "))
        self.assertEqual(cm.exception.error_code, "policy_denied")
        self.pe.check_tool_call("a", 0)
        with self.assertRaises(PolicyDenied):
            self.pe.check_tool_call("a", 1)


class Memory(unittest.TestCase):
    def test_private_memory_stays_with_its_user_and_shared_reaches_everyone(self):
        m = LocalMemory()
        m.save_semantic("active means 90 days", "term:active", "ben", metadata={"match": "active"})
        m.save_taxonomic("d", "churned", "churned means no order in 90 days")
        self.assertEqual(len(m.search("semantic", "active customers", user_id="ben")), 1)
        self.assertEqual(len(m.search("semantic", "active customers", user_id="ana")), 0)
        self.assertEqual(len(m.search("taxonomic", "churned customers", user_id="ana")), 1)

    def test_context_respects_token_budget(self):
        m = LocalMemory()
        m.save_taxonomic("d", "alpha", "alpha " * 400)
        ctx = m.build_context_from_sources("alpha", [SourceSpec("taxonomic")], max_tokens=600)
        self.assertEqual(ctx.chunks, [])            # 600 - 500 reserve cannot fit a 520-token chunk
        ctx = m.build_context_from_sources("alpha", [SourceSpec("taxonomic")], max_tokens=2048)
        self.assertEqual(len(ctx.chunks), 1)


class World(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.cases = load_world()

    def test_pipeline_engine_matches_plain_python(self):
        orders = self.db["orders"]
        july = [o for o in orders if "2026-07-01" <= o["order_date"] < "2026-08-01"]
        m = {"order_date": {"$gte": "2026-07-01", "$lt": "2026-08-01"}}
        rev = aggregate(orders, [{"$match": m}, {"$group": {"_id": None, "value": {"$sum": "$total"}}}])
        self.assertAlmostEqual(rev[0]["value"], sum(o["total"] for o in july), places=2)
        units = aggregate(orders, [{"$match": m}, {"$unwind": "$items"},
                                   {"$group": {"_id": None, "value": {"$sum": "$items.qty"}}}])
        self.assertEqual(units[0]["value"], sum(i["qty"] for o in july for i in o["items"]))
        cnt = aggregate(orders, [{"$match": {"status": {"$ne": "refunded"}}}, {"$count": "value"}])
        self.assertEqual(cnt[0]["value"], sum(1 for o in orders if o["status"] != "refunded"))

    def test_missing_field_sums_to_zero_like_mongodb(self):
        r = aggregate(self.db["orders"], [{"$group": {"_id": None, "value": {"$sum": "$amount"}}}])
        self.assertEqual(r[0]["value"], 0)

    def test_every_guess_gives_a_different_answer_than_the_real_meaning(self):
        for term, per_user in DEFINITIONS.items():
            coll = "orders" if term in ("net revenue", "enterprise") else "customers"
            naive = sum(1 for d in self.db[coll] if matches(d, NAIVE_READING[term]))
            for user, (_, spec) in per_user.items():
                true = sum(1 for d in self.db[coll] if matches(d, spec))
                self.assertNotEqual(naive, true, (term, user))

    def test_no_expected_answer_is_degenerate(self):
        for c in self.cases:
            if isinstance(c.expected, list):
                self.assertTrue(all(v > 0 for v in c.expected), c.id)

    def test_dataset_is_deterministic(self):
        self.assertEqual(build_dataset(7), build_dataset(7))

    def test_writes_never_touch_the_shared_data(self):
        before = len(self.db["orders"])
        t = LocalTools(self.db)
        t.delete_documents("orders", {"is_test": True})
        self.assertGreater(t.mutations, 0)
        self.assertEqual(len(self.db["orders"]), before)


class Loop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = run_ratchet()

    def test_it_improves_and_ends_safe(self):
        v0, final = self.log["versions"][0]["holdout"], self.log["final"]["holdout"]
        self.assertGreater(final["passed"], v0["passed"])
        self.assertGreater(v0["unsafe_runs"], 0)
        self.assertEqual(final["unsafe_runs"], 0)

    def test_the_ratchet_never_slips(self):
        passed = [v["holdout"]["passed"] for v in self.log["versions"]]
        self.assertEqual(passed, sorted(passed))
        for v in self.log["versions"][1:]:
            self.assertTrue(v["evidence"]["fixed"])
            self.assertEqual(v["evidence"]["regressed"], [])

    def test_all_three_outcomes_occur(self):
        decisions = {a["decision"] for a in self.log["attempts"]}
        self.assertEqual(decisions, {"accepted", "rejected_by_gate", "rejected_by_org_ceiling"})

    def test_rejected_patches_change_nothing(self):
        for a in self.log["attempts"]:
            if a["decision"] != "accepted":
                self.assertEqual(a["version_after"], a["from_version"])

    def test_org_ceiling_holds(self):
        self.assertEqual(self.log["org_policy"], ORG_POLICY)
        for v in self.log["versions"]:
            eff = v["effective_policy"]
            self.assertNotIn("export_csv", eff["AUTHORIZED_TOOLS"])
            self.assertLessEqual(eff["MAX_TOOL_CALLS_PER_EXECUTION"], ORG_POLICY["MAX_TOOL_CALLS_PER_EXECUTION"])
        self.assertTrue(self.log["escalations"])

    def test_run_is_deterministic(self):
        again = run_ratchet()
        strip = lambda l: [(a["patch"]["title"], a["decision"]) for a in l["attempts"]]
        self.assertEqual(strip(self.log), strip(again))

    def test_any_version_can_be_rebuilt_from_its_patches(self):
        db, cases = load_world()
        holdout = [c for c in cases if c.split == "holdout"]
        schemas = {k: schema_of(v) for k, v in db.items()}
        patches = self.log["final"]["accepted_patches"]
        for v in self.log["versions"]:
            state = replay(patches, upto=v["version"])
            res = evaluate(state, holdout, db, ScriptedAnalyst(), schemas, False)
            self.assertEqual(res["summary"], v["holdout"], "version %d" % v["version"])
            self.assertEqual(state.harness.to_dict(), v["harness"])


if __name__ == "__main__":
    unittest.main()


# ---- adapters: everything except the network calls themselves ----

from ratchet.core import State
from ratchet.live.agent_engine import StagedMemory
from ratchet.live.llm import LLMAnalyst, LLMEditor, extract_json


class StagedMemoryTests(unittest.TestCase):
    def test_same_story_when_memory_writes_are_staged_then_promoted(self):
        base = LocalMemory()
        plain = run_ratchet()
        staged = run_ratchet(memory=StagedMemory(base))
        strip = lambda l: [(a["patch"]["title"], a["decision"]) for a in l["attempts"]]
        self.assertEqual(strip(plain), strip(staged))
        self.assertEqual(plain["final"]["holdout"], staged["final"]["holdout"])

    def test_rejected_candidates_never_reach_real_memory(self):
        base = LocalMemory()
        run_ratchet(memory=StagedMemory(base))
        active = base.get_taxonomic_term("sales-analytics", "active")
        self.assertIn("30 days", active.text)     # ben's rejected redefinition (90 days) was never written
        self.assertEqual(len([r for r in base.records if r.type == "semantic" and r.user_id == "ben"]), 1)


class ModelAdapters(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(extract_json('Sure:\n```json\n{"a": {"b": "}"}}\n``` done'), {"a": {"b": "}"}})
        self.assertIsNone(extract_json("no json here"))

    def test_analyst_tool_loop_with_a_fake_model(self):
        db, cases = load_world()
        case = next(c for c in cases if c.id == "rev-month-1")
        replies = iter([
            '{"tool": "get_schema", "args": {"collection": "orders"}}',
            '{"tool": "run_query", "args": {"collection": "orders", "pipeline": [{"$match": {"order_date": '
            '{"$gte": "2026-05-01", "$lt": "2026-06-01"}}}, {"$group": {"_id": null, "value": {"$sum": "$total"}}}]}}',
            '{"answer": [%s]}' % case.expected[0],
        ])
        from ratchet.agents import run_case
        from ratchet.loop import seed_state
        state = seed_state()
        state.policy.set_project("MAX_TOOL_CALLS_PER_EXECUTION", 3)
        trace = run_case(state, case, LocalTools(db), LLMAnalyst(lambda s, u: next(replies)))
        self.assertTrue(trace["passed"])
        self.assertEqual([c["tool"] for c in trace["calls"]], ["get_schema", "run_query"])

    def test_analyst_is_halted_by_policy(self):
        db, cases = load_world()
        case = next(c for c in cases if c.id == "rev-month-1")
        from ratchet.agents import run_case
        from ratchet.loop import seed_state
        state = seed_state()
        state.policy.set_project("MAX_TOOL_CALLS_PER_EXECUTION", 1)
        always = lambda s, u: '{"tool": "get_schema", "args": {"collection": "orders"}}'
        trace = run_case(state, case, LocalTools(db), LLMAnalyst(always))
        self.assertFalse(trace["passed"])
        self.assertEqual(trace["denied"]["policy_type"], "MAX_TOOL_CALLS_PER_EXECUTION")

    def test_editor_accepts_valid_changes_and_ignores_malformed_ones(self):
        from ratchet.loop import seed_state
        good = ('{"title": "Revoke delete", "target": "tool access", "rationale": "r", "addresses": "unsafe_write",'
                ' "ops": [{"op": "set_policy", "type": "AUTHORIZED_TOOLS", "value": ["get_schema", "run_query"]}]}')
        patch = LLMEditor(lambda s, u: good).propose(seed_state(), [], [])
        self.assertEqual(patch.ops[0]["type"], "AUTHORIZED_TOOLS")
        for bad in ('{"ops": [{"op": "rm_rf"}]}', '{"ops": [{"op": "set_policy"}]}', '{"stop": true}', "nonsense"):
            self.assertIsNone(LLMEditor(lambda s, u, b=bad: b).propose(seed_state(), [], []))

    def test_a_model_proposed_change_still_cannot_pass_the_org_ceiling(self):
        from ratchet.loop import seed_state
        from ratchet.core import PolicyRejected
        wide = ('{"title": "Grant export", "target": "tool access", "rationale": "r", "addresses": "x", "ops": '
                '[{"op": "set_policy", "type": "AUTHORIZED_TOOLS", "value": ["get_schema", "run_query", "export_csv"]}]}')
        state = seed_state()
        patch = LLMEditor(lambda s, u: wide).propose(state, [], [])
        with self.assertRaises(PolicyRejected):
            state.fork().apply(patch)
