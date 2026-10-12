"""The ratchet: propose -> ceiling check -> gate on held-out cases -> commit or reject."""
from __future__ import annotations

import copy
import datetime as _dt
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from .agents import EDITOR_ID, ScriptedAnalyst, ScriptedEditor, diagnose, run_case
from .core import Harness, LocalMemory, Patch, PolicyEngine, PolicyRejected, State
from .world import ALL_TOOLS, Case, DEFINITIONS, DOMAIN, LocalTools, load_world, schema_of

# The organization's policy. Ratchet can never exceed this; only an org admin can change it.
ORG_POLICY = {
    "AUTHORIZED_TOOLS": ["get_schema", "run_query", "delete_documents"],   # no export_csv
    "MAX_TOOL_CALLS_PER_EXECUTION": 4,
}


def seed_state(memory=None) -> State:
    """Version 0: what a team would plausibly start with."""
    memory = memory if memory is not None else LocalMemory()
    for term in ("net revenue", "enterprise"):       # the org glossary already exists...
        text, spec = DEFINITIONS[term]["*"]
        memory.save_taxonomic(DOMAIN, term, text, metadata={"spec": spec, "source_user": "glossary"})
    harness = Harness(
        version=0,
        context_policy={                             # ...but the default context policy never reads it
            "sources": {"semantic": {"mode": "hybrid", "top_k": 5},
                        "episodic": {"mode": "hybrid", "top_k": 5}},
            "pinned_procedures": [], "max_tokens": 2048},
        tools_offered=["get_schema", "run_query", "delete_documents"],
        project_policies={"MAX_TOOL_CALLS_PER_EXECUTION": 2})
    policy = PolicyEngine(org=ORG_POLICY, project=harness.project_policies)
    return State(harness, memory, policy)


def evaluate(state: State, cases: List[Case], db, analyst, schemas, with_feedback: bool,
             tools_factory: Callable = LocalTools) -> Dict[str, Any]:
    traces = {}
    for case in cases:
        trace = run_case(state, case, tools_factory(db), analyst)
        trace["diagnosis"] = diagnose(trace, case, schemas, with_feedback)
        traces[case.id] = trace
    n = len(cases)
    passed = sum(t["passed"] for t in traces.values())
    return {
        "traces": traces,
        "summary": {
            "passed": passed, "total": n, "pass_rate": round(passed / n, 4) if n else 0.0,
            "unsafe_runs": sum(1 for t in traces.values() if t["mutations"]),
            "policy_denials": sum(1 for t in traces.values() if t["denied"]),
            "avg_tool_calls": round(sum(len(t["calls"]) for t in traces.values()) / n, 2) if n else 0.0,
        },
    }


def gate(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Accept only if something got fixed, nothing that worked broke, and safety did not get worse."""
    b, a = before["traces"], after["traces"]
    fixed = sorted(c for c in b if not b[c]["passed"] and a[c]["passed"])
    regressed = sorted(c for c in b if b[c]["passed"] and not a[c]["passed"])
    less_safe = after["summary"]["unsafe_runs"] > before["summary"]["unsafe_runs"]
    if regressed:
        reason = "broke %d held-out case(s) that were passing: %s" % (len(regressed), ", ".join(regressed))
    elif less_safe:
        reason = "more unsafe runs than before"
    elif not fixed:
        reason = "no held-out case improved"
    else:
        reason = "fixed %d held-out case(s) with no regressions: %s" % (len(fixed), ", ".join(fixed))
    return {"accept": bool(fixed) and not regressed and not less_safe, "reason": reason,
            "fixed": fixed, "regressed": regressed,
            "regression_diagnoses": [a[c]["diagnosis"] for c in regressed]}


def run_ratchet(max_attempts: int = 20, seed: int = 7, analyst=None, editor=None,
                on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                store=None, tools_factory: Callable = LocalTools, memory=None) -> Dict[str, Any]:
    """Run the loop to convergence and return the full, replayable log."""
    emit = on_event or (lambda kind, data: None)
    analyst, editor = analyst or ScriptedAnalyst(), editor or ScriptedEditor()
    db, cases = load_world(seed)
    schemas = {name: schema_of(docs) for name, docs in db.items()}
    train = [c for c in cases if c.split == "train"]
    holdout = [c for c in cases if c.split == "holdout"]
    by_id = {c.id: c for c in cases}

    state = seed_state(memory)
    getattr(state.memory, "commit", lambda: None)()   # the starting glossary is not a candidate
    hold = evaluate(state, holdout, db, analyst, schemas, False, tools_factory)
    tr = evaluate(state, train, db, analyst, schemas, True, tools_factory)
    log: Dict[str, Any] = {
        "meta": {"started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "seed": seed, "analyst": analyst.name, "editor": editor.name,
                 "train_cases": len(train), "holdout_cases": len(holdout),
                 "dataset": {k: len(v) for k, v in db.items()}},
        "org_policy": copy.deepcopy(ORG_POLICY),
        "versions": [{"version": 0, "parent": None, "patch": None, "harness": state.harness.to_dict(),
                      "effective_policy": state.policy.effective(),
                      "holdout": hold["summary"], "train": tr["summary"]}],
        "attempts": [], "escalations": [],
    }
    emit("baseline", log["versions"][0])
    blocked: List[str] = []
    accepted_patches: List[Dict[str, Any]] = []

    for n in range(1, max_attempts + 1):
        failures = [{"case": by_id[cid], "trace": t, "diagnosis": t["diagnosis"]}
                    for cid, t in tr["traces"].items() if not t["passed"]]
        patch = editor.propose(state, failures, blocked)
        if patch is None:
            break
        attempt: Dict[str, Any] = {"n": n, "from_version": state.harness.version, "patch": patch.to_dict(),
                                   "holdout_before": hold["summary"], "train_before": tr["summary"]}
        candidate = state.fork()
        try:
            candidate.apply(patch)
        except PolicyRejected as e:
            # The platform refuses the write. Nothing changes; a human has to decide.
            attempt.update(decision="rejected_by_org_ceiling", reason=str(e), fixed=[], regressed=[],
                           regression_diagnoses=[], holdout_after=None, version_after=state.harness.version)
            blocked.append(patch.addresses)
            log["escalations"].append({"attempt": n, "patch": patch.title, "reason": str(e),
                                       "needs": "organization admin decision"})
        else:
            cand_hold = evaluate(candidate, holdout, db, analyst, schemas, False, tools_factory)
            verdict = gate(hold, cand_hold)
            attempt.update(reason=verdict["reason"], fixed=verdict["fixed"], regressed=verdict["regressed"],
                           regression_diagnoses=verdict["regression_diagnoses"],
                           holdout_after=cand_hold["summary"])
            if verdict["accept"]:
                candidate.harness.parent = state.harness.version
                candidate.harness.version = state.harness.version + 1
                state, hold = candidate, cand_hold
                getattr(state.memory, "commit", lambda: None)()   # promote staged memory writes
                tr = evaluate(state, train, db, analyst, schemas, True, tools_factory)
                accepted_patches.append(patch.to_dict())
                attempt.update(decision="accepted", version_after=state.harness.version)
                log["versions"].append({
                    "version": state.harness.version, "parent": state.harness.parent,
                    "patch": patch.to_dict(), "harness": state.harness.to_dict(),
                    "effective_policy": state.policy.effective(),
                    "holdout": hold["summary"], "train": tr["summary"], "evidence": verdict})
            else:
                attempt.update(decision="rejected_by_gate", version_after=state.harness.version)
        # The editor's own record of what it tried and what happened (episodic memory).
        state.memory.save_episode(
            title="Attempt %d: %s" % (n, patch.title),
            content="%s. Outcome: %s. %s" % (patch.rationale, attempt["decision"], attempt["reason"]),
            user_id=EDITOR_ID,
            metadata={"signature": patch.signature(), "decision": attempt["decision"],
                      "reason": attempt["reason"], "patch": patch.to_dict(),
                      "regression_diagnoses": attempt["regression_diagnoses"]})
        log["attempts"].append(attempt)
        if store is not None:
            store.save_attempt(attempt)
            if attempt["decision"] == "accepted":
                store.save_version(log["versions"][-1])
        emit("attempt", attempt)

    remaining = [{"case_id": cid, "question": t["question"], "diagnosis": t["diagnosis"]}
                 for part in (tr, hold) for cid, t in part["traces"].items() if not t["passed"]]
    log["final"] = {"version": state.harness.version, "harness": state.harness.to_dict(),
                    "effective_policy": state.policy.effective(), "holdout": hold["summary"],
                    "train": tr["summary"], "remaining_failures": remaining,
                    "memory": [{"type": r.type, "key": r.key, "text": r.text, "user_id": r.user_id,
                                "visibility": r.visibility} for r in getattr(state.memory, "records", [])
                               if r.user_id != EDITOR_ID],
                    "accepted_patches": accepted_patches}
    emit("final", log["final"])
    return log


def replay(patches: List[Dict[str, Any]], upto: Optional[int] = None) -> State:
    """Rebuild any version from the seed by re-applying accepted patches. This is rollback."""
    state = seed_state()
    for p in patches[: len(patches) if upto is None else upto]:
        state.apply(Patch(**p))
        state.harness.parent, state.harness.version = state.harness.version, state.harness.version + 1
    return state
