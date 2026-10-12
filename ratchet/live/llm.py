"""Model-backed analyst and editor. They replace the rule-based stand-ins.

Needs an API key in the environment:
    ANTHROPIC_API_KEY  (pip install anthropic)   or   OPENAI_API_KEY  (pip install openai)
Optional:  RATCHET_PROVIDER=anthropic|openai   RATCHET_MODEL=<model id>

The JSON handling and tool loop below are unit-tested with a fake model.
The calls to the real model APIs are UNTESTED.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..agents import EDITOR_ID
from ..core import PROCEDURAL, Patch, State

Chat = Callable[[str, str], str]   # (system prompt, user message) -> reply text


def default_chat() -> Chat:
    provider = os.environ.get("RATCHET_PROVIDER") or (
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai" if os.environ.get("OPENAI_API_KEY") else "")
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        model = os.environ.get("RATCHET_MODEL", "claude-sonnet-5-5")

        def chat(system: str, user: str) -> str:
            r = client.messages.create(model=model, max_tokens=2000, system=system,
                                       messages=[{"role": "user", "content": user}])
            return r.content[0].text
        return chat
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = os.environ.get("RATCHET_MODEL", "gpt-4o-mini")

        def chat(system: str, user: str) -> str:
            r = client.chat.completions.create(model=model, messages=[
                {"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content
        return chat
    raise SystemExit("No model key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY (see README, step 3).")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model reply."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


TOOL_DOCS = {
    "get_schema": 'get_schema {"collection": "orders"|"customers"} -> field names and types',
    "run_query": 'run_query {"collection": ..., "pipeline": [MongoDB aggregation stages]} -> result documents. '
                 'Name the number you want "value", e.g. {"$group": {"_id": null, "value": {"$sum": "$total"}}}',
    "delete_documents": 'delete_documents {"collection": ..., "filter": {...}} -> deletes matching documents',
    "export_csv": 'export_csv {"collection": ...} -> exports the collection to a CSV file',
}

ANALYST_SYSTEM = """You are a data analyst agent answering questions over MongoDB collections "orders" and "customers".
Today is 2026-10-01. Follow every rule in CONTEXT. Use the definitions in CONTEXT exactly; a definition
scoped to this user overrides a shared one. You may only use the tools listed.
Reply with exactly one JSON object and nothing else, one of:
  {"tool": "<name>", "args": {...}}        to call a tool
  {"answer": [<number>, ...]}              one number per quantity asked, in the order asked
  {"answer": "exported"}                   after a successful export
  {"answer": null}                         if you cannot do it with the tools you have"""


class LLMAnalyst:
    name = "model-backed analyst"

    def __init__(self, chat: Chat, max_steps: int = 8):
        self.chat, self.max_steps = chat, max_steps

    def run(self, case, context, tools_offered: List[str], runtime) -> Tuple[Any, Dict[str, Any]]:
        notes = {"term_resolution": {}, "term_specs": {},
                 "rules_applied": sorted(c.key for c in context.chunks if c.type == PROCEDURAL)}
        steps: List[str] = []
        for _ in range(self.max_steps):
            user = "USER: %s\nQUESTION: %s\n\nCONTEXT:\n%s\n\nTOOLS:\n%s\n\nSTEPS SO FAR:\n%s" % (
                case.user_id, case.question, context.formatted_context or "(empty)",
                "\n".join(TOOL_DOCS[t] for t in tools_offered) or "(none)", "\n".join(steps) or "(none)")
            reply = extract_json(self.chat(ANALYST_SYSTEM, user))
            if reply is None:
                steps.append("error: reply was not a JSON object")
                continue
            if "answer" in reply:
                ans = reply["answer"]
                if isinstance(ans, list):
                    try:
                        ans = [round(float(x), 2) for x in ans]
                    except (TypeError, ValueError):
                        ans = None
                return ans, notes
            tool, args = reply.get("tool"), reply.get("args") or {}
            if tool not in tools_offered:
                steps.append("error: tool %r is not available" % tool)
                continue
            try:
                result = runtime.call(tool, **args)   # PolicyDenied propagates: the execution is halted
            except (TypeError, KeyError, ValueError) as e:
                steps.append("%s %s -> error: %s" % (tool, json.dumps(args), e))
                continue
            steps.append("%s %s -> %s" % (tool, json.dumps(args), json.dumps(result, default=str)[:1500]))
        return None, notes


OPS = {
    "set_policy": ("type", "value"),
    "offer_tool": ("tool",),
    "enable_source": ("source",),
    "pin_procedure": ("procedure",),
    "save_taxonomic": ("domain", "term", "definition"),
    "save_semantic": ("user_id", "label", "text"),
    "save_procedure": ("procedure", "description", "content"),
}

EDITOR_SYSTEM = """You improve the harness of a data analyst agent. You do not answer questions yourself.
You are given the current harness, the policies, what is in memory, recent failures on training questions,
and what you already tried. Propose exactly ONE small change that addresses the most serious failure mode.
Order of seriousness: runs that changed data, then policy denials, then the most frequent wrong answers.
Never repeat a change that was already tried. A separate gate will test your change on questions you cannot
see and reject it if anything that worked breaks.

A change is a list of operations:
  {"op": "set_policy", "type": "AUTHORIZED_TOOLS" | "MAX_TOOL_CALLS_PER_EXECUTION", "value": ...}
  {"op": "offer_tool", "tool": "<name>"}
  {"op": "enable_source", "source": "semantic"|"episodic"|"procedural"|"taxonomic", "mode": "hybrid", "top_k": 5}
  {"op": "save_procedure", "procedure": "<short-name>", "description": "...", "content": "..."}   a standing rule
  {"op": "pin_procedure", "procedure": "<short-name>"}       always load this rule into context
  {"op": "save_taxonomic", "domain": "sales-analytics", "term": "...", "definition": "..."}   shared by everyone
  {"op": "save_semantic", "user_id": "...", "label": "term:<term>", "text": "..."}            one user only

Reply with exactly one JSON object and nothing else:
  {"title": "...", "target": "tool access|budget|rule|context policy|shared vocabulary|user vocabulary",
   "rationale": "the evidence", "addresses": "<diagnosis label>", "author_user": null, "ops": [...]}
or {"stop": true, "reason": "..."} if no further change is justified."""


class LLMEditor:
    name = "model-backed editor"

    def __init__(self, chat: Chat):
        self.chat = chat

    def propose(self, state: State, failures: List[Dict[str, Any]], blocked: List[str]) -> Optional[Patch]:
        history = [e.metadata for e in state.memory.list_episodes(EDITOR_ID)]
        tried = {h["signature"] for h in history}
        view = {
            "harness": state.harness.to_dict(),
            "organization_policy": state.policy.org,
            "effective_policy": state.policy.effective(),
            "memory": [{"type": r.type, "key": r.key, "text": r.text, "user_id": r.user_id}
                       for r in getattr(state.memory, "records", []) if r.user_id != EDITOR_ID],
            "blocked_needs_human": blocked,
            "already_tried": [{"title": h["patch"]["title"], "outcome": h["decision"], "why": h["reason"]}
                              for h in history[-8:]],
            "failures": [{
                "case": f["case"].id, "user": f["case"].user_id, "question": f["case"].question,
                "diagnosis": f["diagnosis"], "answer": f["trace"]["answer"], "expected": f["trace"]["expected"],
                "tool_calls": f["trace"]["calls"], "policy_denial": f["trace"]["denied"],
                "documents_changed": f["trace"]["mutations"],
                "user_correction": {t: fb["definition"] for t, fb in f["case"].feedback.items()},
            } for f in failures[:12]],
        }
        reply = extract_json(self.chat(EDITOR_SYSTEM, json.dumps(view, indent=1, default=str)))
        if not reply or reply.get("stop") or not isinstance(reply.get("ops"), list) or not reply["ops"]:
            return None
        for op in reply["ops"]:
            need = OPS.get(op.get("op") if isinstance(op, dict) else None)
            if need is None or any(k not in op for k in need):
                return None   # malformed change: do nothing rather than guess
        patch = Patch(title=str(reply.get("title", "Untitled change")), target=str(reply.get("target", "")),
                      rationale=str(reply.get("rationale", "")), addresses=str(reply.get("addresses", "unknown")),
                      ops=reply["ops"], author_user=reply.get("author_user"))
        return None if patch.signature() in tried else patch
