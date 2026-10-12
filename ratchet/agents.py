"""The two agents.

  Analyst: the working agent. Answers a question using only what the harness
           gives it (context, tools, budget).
  Editor:  the agent that changes the harness. Reads failures, proposes one
           patch at a time.

Offline, both are rule-based stand-ins for a language model so the loop runs
with no accounts or keys. They are deliberately limited: the analyst knows how
to read a question's structure but does not know your jargon, your schema, or
that deleting data is a bad idea. Everything it gets right beyond that comes
from the harness. ratchet/live/llm.py has the model-backed versions.
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .core import (PolicyDenied, Patch, State, SourceSpec, PROCEDURAL, SEMANTIC, TAXONOMIC)
from .world import (Case, DOMAIN, WRITE_TOOLS, compile_queries, pipeline_fields, read_value, schema_of)

EDITOR_ID = "ratchet-editor"
SCHEMA_RULE = "schema-first"

# What the analyst assumes when nobody has told it what a term means.
NAIVE_READING = {
    "net revenue": {},                                                  # same as revenue
    "enterprise": {"customer_tier": {"$in": ["enterprise", "business"]}},  # any company account
    "churned": {"status": "inactive"},                                  # trusts the CRM flag
    "active": {"status": "active"},
}
GUESSED_FIELDS = {"region": "region", "qty": "quantity"}


class Runtime:
    """Routes every tool call through the policy checkpoint and records a trace."""

    def __init__(self, policy, tools):
        self.policy, self.tools = policy, tools
        self.calls: List[Dict[str, Any]] = []
        self.denied: Optional[Dict[str, str]] = None

    def call(self, tool: str, **args):
        try:
            self.policy.check_tool_call(tool, len(self.calls))
        except PolicyDenied as e:
            self.denied = {"policy_type": e.policy_type, "message": str(e), "error_code": e.error_code}
            raise
        result = getattr(self.tools, tool)(**args)
        self.calls.append({"tool": tool, "args": copy.deepcopy(args)})
        return result


class ScriptedAnalyst:
    name = "scripted-analyst (offline stand-in for a language model)"

    def run(self, case: Case, context, tools_offered: List[str], runtime: Runtime) -> Tuple[Any, Dict[str, Any]]:
        intent = case.intent
        coll = intent["collection"]
        rules = {c.key for c in context.chunks if c.type == PROCEDURAL}
        notes: Dict[str, Any] = {"term_resolution": {}, "term_specs": {}, "rules_applied": sorted(rules)}

        # Field names: guessed, unless a rule in context says to check the schema first.
        fields = dict(GUESSED_FIELDS)
        if SCHEMA_RULE in rules and "get_schema" in tools_offered:
            schema = runtime.call("get_schema", collection=coll)
            region = [k for k in schema if "region" in k]
            fields["region"] = region[0] if region else fields["region"]
            fields["qty"] = "qty" if "items.qty" in schema else fields["qty"]

        # Jargon: a definition scoped to this user wins over a shared one; else guess.
        specs: Dict[str, Dict[str, Any]] = {}
        for term in intent.get("terms", []):
            user_def = [c for c in context.chunks if c.type == SEMANTIC and c.metadata.get("term") == term]
            shared_def = [c for c in context.chunks if c.type == TAXONOMIC and c.key == term]
            if user_def:
                specs[term], notes["term_resolution"][term] = user_def[0].metadata["spec"], "user"
            elif shared_def:
                specs[term], notes["term_resolution"][term] = shared_def[0].metadata["spec"], "shared"
            else:
                specs[term], notes["term_resolution"][term] = NAIVE_READING[term], "naive"
            notes["term_specs"][term] = specs[term]

        action = intent.get("action")
        if action == "export":
            if "export_csv" not in tools_offered:
                return None, notes  # no way to do it
            runtime.call("export_csv", collection=coll)
            return "exported", notes

        exclude_test = False
        if action == "cleanup":
            if "delete_documents" in tools_offered:   # asked to delete, has the tool: deletes
                runtime.call("delete_documents", collection=coll, filter={"is_test": True})
            else:                                     # no write tool: filters instead
                exclude_test = True

        answer = []
        for c, pipeline in compile_queries(intent, specs, fields, exclude_test=exclude_test):
            answer.append(read_value(runtime.call("run_query", collection=c, pipeline=pipeline)))
        return answer, notes


def offered_tools(state: State) -> List[str]:
    """Tools the agent is given: what the harness registers, minus what policy forbids."""
    allowed = state.policy.effective().get("AUTHORIZED_TOOLS")
    return [t for t in state.harness.tools_offered if allowed is None or t in allowed]


def build_context(state: State, case: Case):
    cp = state.harness.context_policy
    sources = [SourceSpec(name, s.get("mode", "hybrid"), s.get("top_k", 5))
               for name, s in cp.get("sources", {}).items()]
    return state.memory.build_context_from_sources(
        case.question, sources, user_id=case.user_id, max_tokens=cp.get("max_tokens", 2048),
        pinned_procedures=cp.get("pinned_procedures", []))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(abs(x - y) < 0.01 for x, y in zip(a, b))
    return a == b


def run_case(state: State, case: Case, tools, analyst) -> Dict[str, Any]:
    """One execution of the working agent under the given harness. Returns its trace."""
    context = build_context(state, case)
    runtime = Runtime(state.policy, tools)
    answer, notes = None, {"term_resolution": {}, "rules_applied": []}
    try:
        answer, notes = analyst.run(case, context, offered_tools(state), runtime)
    except PolicyDenied:
        pass  # the platform halts the execution on a denial
    mutations = getattr(tools, "mutations", 0)
    passed = runtime.denied is None and mutations == 0 and _same(answer, case.expected)
    return {
        "case_id": case.id, "user_id": case.user_id, "question": case.question,
        "context": [{"type": c.type, "key": c.key, "scope": c.scope} for c in context.chunks],
        "context_tokens": context.token_count,
        "term_resolution": notes.get("term_resolution", {}),
        "term_specs": notes.get("term_specs", {}),
        "rules_applied": notes.get("rules_applied", []),
        "calls": runtime.calls, "denied": runtime.denied, "mutations": mutations,
        "answer": answer, "expected": case.expected, "passed": passed,
    }


# --------------------------------------------------------------------------
# Diagnosis: turn a failed trace into a named failure mode
# --------------------------------------------------------------------------

def diagnose(trace: Dict[str, Any], case: Case, schemas: Dict[str, Dict[str, str]],
             with_feedback: bool) -> str:
    """Name the failure. Uses the trace and the schema; user feedback only on training cases."""
    if trace["mutations"]:
        return "unsafe_write"
    if trace["denied"]:
        return "policy_denied:" + trace["denied"]["policy_type"]
    if trace["passed"]:
        return "ok"
    for call in trace["calls"]:
        if call["tool"] == "run_query":
            known = schemas[call["args"]["collection"]]
            if any(f not in known for f in pipeline_fields(call["args"]["pipeline"])):
                return "wrong_field"
    if case.intent.get("action") == "export":
        return "missing_capability:export_csv"
    resolution = trace["term_resolution"]
    for term, how in resolution.items():          # a guessed term is the first suspect
        if how == "naive":
            return "undefined_term:" + term
    for term, how in resolution.items():
        if how != "shared":
            continue
        if not with_feedback:                      # held-out case: we only know a shared meaning was used
            return "shared_term:" + term
        if case.feedback.get(term, {}).get("spec") != trace["term_specs"].get(term):
            return "term_conflict:" + term         # the user's correction differs from the shared meaning
    return "unknown"


# --------------------------------------------------------------------------
# Editor
# --------------------------------------------------------------------------

class ScriptedEditor:
    """Proposes one harness patch per round, most serious failure mode first."""

    name = "scripted-editor (offline stand-in for a language model)"

    def propose(self, state: State, failures: List[Dict[str, Any]], blocked: List[str]) -> Optional[Patch]:
        history = [e.metadata for e in state.memory.list_episodes(EDITOR_ID)]  # its own episodic memory
        tried = {h["signature"] for h in history}

        # 1. Repair: if the gate just rejected a patch for a side effect we know how
        #    to fix, carry that fix along with the original patch.
        if history and history[-1]["decision"] == "rejected_by_gate" and history[-1]["regression_diagnoses"]:
            last = history[-1]
            side_effect = Counter(last["regression_diagnoses"]).most_common(1)[0][0]
            for fix in self._candidates(side_effect, state, []):
                combo = Patch(
                    title="%s + %s" % (last["patch"]["title"], fix.title[0].lower() + fix.title[1:]),
                    target="%s + %s" % (last["patch"]["target"], fix.target),
                    rationale="Last attempt was rejected: %s. This carries the fix for that side effect."
                              % last["reason"],
                    addresses=last["patch"]["addresses"],
                    ops=last["patch"]["ops"] + fix.ops,
                    author_user=last["patch"].get("author_user"))
                if combo.signature() not in tried:
                    return combo

        # 2. Otherwise: safety first, then policy denials, then the most common failure.
        counts = Counter(f["diagnosis"] for f in failures)

        options = {d: self._candidates(d, state, [f for f in failures if f["diagnosis"] == d]) for d in counts}

        def rank(d: str):
            severity = 0 if d == "unsafe_write" else 1 if d.startswith("policy_denied") else 2
            size = len(options[d][0].ops) if options[d] else 99   # equal evidence: smaller change first
            return (severity, -counts[d], size, d)

        for diag in sorted(counts, key=rank):
            if diag in blocked:
                continue
            for patch in options[diag]:
                if patch.signature() not in tried:
                    return patch
        return None

    def _candidates(self, diag: str, state: State, group: List[Dict[str, Any]]) -> List[Patch]:
        eff = state.policy.effective()
        n = len(group)
        ids = ", ".join(f["case"].id for f in group)
        sources = state.harness.context_policy.get("sources", {})

        if diag == "unsafe_write":
            used = sorted({c["tool"] for f in group for c in f["trace"]["calls"] if c["tool"] in WRITE_TOOLS})
            allowed = [t for t in eff.get("AUTHORIZED_TOOLS", state.harness.tools_offered) if t not in used]
            return [Patch(
                title="Revoke %s" % ", ".join(used), target="tool access",
                rationale="%d training run(s) changed stored data (%s). Removing the write tool makes that "
                          "impossible instead of relying on the agent to behave." % (n, ids),
                addresses=diag, ops=[{"op": "set_policy", "type": "AUTHORIZED_TOOLS", "value": allowed}])]

        if diag == "policy_denied:MAX_TOOL_CALLS_PER_EXECUTION":
            cap = eff["MAX_TOOL_CALLS_PER_EXECUTION"]
            return [Patch(
                title="Raise tool-call cap to %d" % (cap + 1), target="budget",
                rationale="Runs were halted at the cap of %d tool calls. Raising it by one, the smallest step."
                          % cap,
                addresses=diag,
                ops=[{"op": "set_policy", "type": "MAX_TOOL_CALLS_PER_EXECUTION", "value": cap + 1}])]

        if diag == "wrong_field":
            return [Patch(
                title="Add rule: check the schema before querying", target="rule",
                rationale="%d training run(s) queried fields that do not exist in the collection (%s), "
                          "which silently returns zero." % (n, ids),
                addresses=diag,
                ops=[{"op": "save_procedure", "procedure": SCHEMA_RULE,
                      "description": "Check the collection schema before writing a query.",
                      "content": "Before any run_query call, call get_schema on the collection and use "
                                 "only field names it returns."},
                     {"op": "pin_procedure", "procedure": SCHEMA_RULE}])]

        if diag.startswith("undefined_term:"):
            term = diag.split(":", 1)[1]
            existing = state.memory.get_taxonomic_term(DOMAIN, term)
            enable = {"op": "enable_source", "source": TAXONOMIC, "mode": "hybrid", "top_k": 5}
            if existing is not None and TAXONOMIC not in sources:
                return [Patch(
                    title="Context policy: retrieve taxonomic memory", target="context policy",
                    rationale="%d training run(s) guessed the meaning of '%s' (%s). The definition is already "
                              "in shared memory, but the context policy never retrieves taxonomic memory."
                              % (n, term, ids),
                    addresses=diag, ops=[enable])]
            if existing is None and group:
                f = group[0]
                fb = f["case"].feedback[term]
                ops = ([] if TAXONOMIC in sources else [enable]) + [{
                    "op": "save_taxonomic", "domain": DOMAIN, "term": term, "definition": fb["definition"],
                    "metadata": {"spec": fb["spec"], "source_user": f["case"].user_id}}]
                return [Patch(
                    title="Shared vocabulary: define '%s'" % term, target="shared vocabulary",
                    rationale="%d training run(s) guessed the meaning of '%s' (%s). %s corrected it: \"%s\""
                              % (n, term, ids, f["case"].user_id, fb["definition"]),
                    addresses=diag, ops=ops, author_user=f["case"].user_id)]
            return []

        if diag.startswith("term_conflict:") and group:
            term = diag.split(":", 1)[1]
            f = group[0]
            user, fb = f["case"].user_id, f["case"].feedback[term]
            why = ("%d training run(s) for %s used the shared meaning of '%s' and were wrong (%s). %s says: \"%s\""
                   % (n, user, term, ids, user, fb["definition"]))
            return [
                Patch(title="Shared vocabulary: redefine '%s'" % term, target="shared vocabulary",
                      rationale=why + " First try: update the shared definition.", addresses=diag,
                      ops=[{"op": "save_taxonomic", "domain": DOMAIN, "term": term,
                            "definition": fb["definition"],
                            "metadata": {"spec": fb["spec"], "source_user": user}}], author_user=user),
                Patch(title="User vocabulary: '%s' for %s only" % (term, user), target="user vocabulary",
                      rationale=why + " Scoped to this user so nobody else's meaning changes.", addresses=diag,
                      ops=[{"op": "save_semantic", "user_id": user, "label": "term:" + term,
                            "text": fb["definition"],
                            "metadata": {"term": term, "match": term, "spec": fb["spec"]}}], author_user=user),
            ]

        if diag == "missing_capability:export_csv":
            allowed = list(eff.get("AUTHORIZED_TOOLS", state.harness.tools_offered)) + ["export_csv"]
            return [Patch(
                title="Grant export_csv", target="tool access",
                rationale="%d training run(s) asked for a CSV export the agent has no tool for (%s)." % (n, ids),
                addresses=diag,
                ops=[{"op": "offer_tool", "tool": "export_csv"},
                     {"op": "set_policy", "type": "AUTHORIZED_TOOLS", "value": allowed}])]
        return []
