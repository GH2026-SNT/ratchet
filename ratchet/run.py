"""Command line entry point.

  python -m ratchet.run                    run the loop offline, print the story
  python -m ratchet.run --pause            wait for Enter between attempts (for a live demo)
  python -m ratchet.run --out run.json     also save the full log
  python -m ratchet.run --rollback 2       afterwards, rebuild version 2 and re-score it
  python -m ratchet.run --atlas            record versions in Atlas and run queries there
                                           (needs MONGODB_URI and: pip install pymongo)
  python -m ratchet.run --llm              use a real model for both agents
                                           (needs an API key; see README)
  python -m ratchet.run --agent-engine-memory
                                           keep rules and vocabulary in Atlas Agent Engine memory
                                           (needs the memory service; see README)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .agents import ScriptedAnalyst
from .loop import ORG_POLICY, evaluate, replay, run_ratchet
from .world import LocalTools, load_world, schema_of

LABEL = {"accepted": "ACCEPTED", "rejected_by_gate": "REJECTED by the gate",
         "rejected_by_org_ceiling": "REFUSED by organization policy"}


def _score(s) -> str:
    return "%d/%d held-out cases (%.0f%%), %d unsafe run(s)" % (
        s["passed"], s["total"], 100 * s["pass_rate"], s["unsafe_runs"])


def make_printer(pause: bool):
    def on_event(kind, data):
        if kind == "baseline":
            print("=" * 72)
            print("RATCHET  |  a harness that rewrites itself, inside limits it cannot widen")
            print("=" * 72)
            print("Organization policy (the ceiling): %s" % json.dumps(ORG_POLICY))
            print("\nVersion 0 (starting harness): %s" % _score(data["holdout"]))
            print("  tools offered:   %s" % ", ".join(data["harness"]["tools_offered"]))
            print("  context sources: %s" % ", ".join(data["harness"]["context_policy"]["sources"]))
            print("  tool-call cap:   %s" % data["effective_policy"].get("MAX_TOOL_CALLS_PER_EXECUTION"))
        elif kind == "attempt":
            if pause:
                input("\n[Enter for the next attempt] ")
            p = data["patch"]
            print("\n--- Attempt %d: %s  [%s]" % (data["n"], p["title"], p["target"]))
            print("    why:     %s" % p["rationale"])
            print("    result:  %s" % LABEL[data["decision"]])
            print("    because: %s" % data["reason"])
            if data["decision"] == "accepted":
                print("    now at version %d: %s" % (data["version_after"], _score(data["holdout_after"])))
        elif kind == "final":
            print("\n" + "=" * 72)
            print("Converged at version %d: %s" % (data["version"], _score(data["holdout"])))
            print("  tools allowed:   %s" % ", ".join(data["effective_policy"]["AUTHORIZED_TOOLS"]))
            print("  context sources: %s" % ", ".join(data["harness"]["context_policy"]["sources"]))
            print("  standing rules:  %s" % (", ".join(data["harness"]["context_policy"]["pinned_procedures"]) or "none"))
            print("  tool-call cap:   %s (organization cap: %s)" % (
                data["effective_policy"]["MAX_TOOL_CALLS_PER_EXECUTION"],
                ORG_POLICY["MAX_TOOL_CALLS_PER_EXECUTION"]))
            if data["remaining_failures"]:
                kinds = sorted({r["diagnosis"] for r in data["remaining_failures"]})
                print("  still failing:   %d case(s) across both sets: %s  -> needs a human decision"
                      % (len(data["remaining_failures"]), ", ".join(kinds)))
    return on_event


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ratchet.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the full run log to this JSON file")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between attempts")
    ap.add_argument("--quiet", action="store_true", help="print only the final line")
    ap.add_argument("--rollback", type=int, metavar="N", help="rebuild version N and re-score it")
    ap.add_argument("--atlas", action="store_true", help="use MongoDB Atlas (MONGODB_URI) for data and records")
    ap.add_argument("--llm", action="store_true", help="use a real language model for both agents")
    ap.add_argument("--agent-engine-memory", action="store_true",
                    help="use Atlas Agent Engine memory (changes are staged and promoted only when accepted)")
    ap.add_argument("--max-attempts", type=int, default=20)
    args = ap.parse_args(argv)

    kwargs = {}
    if args.atlas:
        from .live.atlas import AtlasTools, MongoStore, connect, seed_dataset
        mdb = connect(os.environ.get("MONGODB_URI"))
        seed_dataset(mdb)
        kwargs.update(store=MongoStore(mdb), tools_factory=lambda _db: AtlasTools(mdb))
    if args.agent_engine_memory:
        from .live.agent_engine import AgentEngineMemory, StagedMemory
        kwargs.update(memory=StagedMemory(AgentEngineMemory()))
    if args.llm:
        from .live.llm import LLMAnalyst, LLMEditor, default_chat
        chat = default_chat()
        kwargs.update(analyst=LLMAnalyst(chat), editor=LLMEditor(chat))

    log = run_ratchet(max_attempts=args.max_attempts,
                      on_event=None if args.quiet else make_printer(args.pause), **kwargs)
    if args.quiet:
        print("version %d: %s" % (log["final"]["version"], _score(log["final"]["holdout"])))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print("\nFull log written to %s" % args.out)

    if args.rollback is not None:
        db, cases = load_world()
        state = replay(log["final"]["accepted_patches"], upto=args.rollback)
        res = evaluate(state, [c for c in cases if c.split == "holdout"], db, ScriptedAnalyst(),
                       {k: schema_of(v) for k, v in db.items()}, False, LocalTools)
        print("\nRolled back to version %d by replaying its patches: %s" % (args.rollback, _score(res["summary"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
