"""Harness document, policy ceiling, and memory.

This module is the offline reference implementation. Class and method names
mirror MongoDB Atlas Agent Engine so the live adapters in ratchet/live/ can
replace them one for one:

  PolicyEngine  -> Agent Engine Policy Engine (org + project scopes)
  LocalMemory   -> agent_engine_sdk_memory.Memory (semantic, episodic,
                   procedural, taxonomic)

Python 3.9+ and the standard library only.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Policy ceiling
# --------------------------------------------------------------------------

ALLOWLIST, CAP = "allowlist", "cap"

# Policy types and kinds as documented for the Agent Engine Policy Engine.
POLICY_KINDS = {
    "AUTHORIZED_TOOLS": ALLOWLIST,
    "AUTHORIZED_MODELS": ALLOWLIST,
    "MAX_TOOL_CALLS_PER_EXECUTION": CAP,
    "MAX_LLM_CALLS_PER_EXECUTION": CAP,
    "MAX_EXECUTION_DURATION_MS": CAP,
    "MAX_TOKENS_PER_EXECUTION": CAP,
    "MAX_TOKENS_PER_SESSION": CAP,
}


class PolicyRejected(Exception):
    """A project policy write was refused because it is wider than the org policy."""


class PolicyDenied(Exception):
    """An execution was halted at a policy checkpoint."""

    error_code = "policy_denied"

    def __init__(self, policy_type: str, reason: str):
        super().__init__("Policy denied: " + reason)
        self.policy_type = policy_type
        self.reason = reason


class PolicyEngine:
    """Org policy is the ceiling. Project policy can only be equal or tighter.

    Merge rule (same as the platform): allowlists intersect, numeric caps take
    the lower value. A project write that is less restrictive than the org
    policy of the same type is rejected.
    """

    def __init__(self, org: Optional[Dict[str, Any]] = None, project: Optional[Dict[str, Any]] = None):
        self.org: Dict[str, Any] = copy.deepcopy(org or {})
        self.project: Dict[str, Any] = {}
        for ptype, value in (project or {}).items():
            self.set_project(ptype, value)

    def validate_project(self, ptype: str, value: Any) -> None:
        kind = POLICY_KINDS.get(ptype)
        if kind is None:
            raise PolicyRejected("unknown policy type %s" % ptype)
        if ptype not in self.org:
            return
        if kind == ALLOWLIST:
            extra = sorted(set(value) - set(self.org[ptype]))
            if extra:
                raise PolicyRejected(
                    "%s: project cannot add %s; the organization allowlist omits it" % (ptype, ", ".join(extra))
                )
        elif value > self.org[ptype]:
            raise PolicyRejected(
                "%s: project cap %s is higher than the organization cap %s" % (ptype, value, self.org[ptype])
            )

    def set_project(self, ptype: str, value: Any) -> None:
        self.validate_project(ptype, value)
        self.project[ptype] = copy.deepcopy(value)

    def effective(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for ptype, kind in POLICY_KINDS.items():
            o, p = self.org.get(ptype), self.project.get(ptype)
            if o is None and p is None:
                continue  # no policy of this type: unrestricted
            if o is None or p is None:
                out[ptype] = copy.deepcopy(p if o is None else o)
            elif kind == ALLOWLIST:
                out[ptype] = [t for t in o if t in set(p)]
            else:
                out[ptype] = min(o, p)
        return out

    def check_tool_call(self, tool: str, calls_so_far: int) -> None:
        """Pre-tool-call checkpoint."""
        eff = self.effective()
        allowed = eff.get("AUTHORIZED_TOOLS")
        if allowed is not None and tool not in allowed:
            raise PolicyDenied("AUTHORIZED_TOOLS", "tool '%s' is not authorized" % tool)
        cap = eff.get("MAX_TOOL_CALLS_PER_EXECUTION")
        if cap is not None and calls_so_far >= cap:
            raise PolicyDenied(
                "MAX_TOOL_CALLS_PER_EXECUTION", "execution reached the cap of %s tool calls" % cap
            )


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

SEMANTIC, EPISODIC, PROCEDURAL, TAXONOMIC = "semantic", "episodic", "procedural", "taxonomic"
FORMAT_RESERVE_TOKENS = 500  # the platform subtracts a formatting reserve from max_tokens


@dataclass
class Record:
    type: str
    key: str                      # label / term / procedure name / episode title
    text: str
    user_id: Optional[str] = None
    visibility: str = "private"   # "private" (owner only) or "org" (whole project)
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceSpec:
    source: str
    mode: str = "hybrid"
    top_k: int = 5


@dataclass
class Chunk:
    type: str
    key: str
    text: str
    score: float
    scope: str                    # "user" or "shared"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    chunks: List[Chunk]
    formatted_context: str
    token_count: int


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text.split()) * 1.3))


class LocalMemory:
    """In-process stand-in for Agent Engine memory.

    Retrieval here is keyword overlap. The live adapter uses the platform's
    hybrid search with Voyage AI embeddings instead.
    """

    def __init__(self) -> None:
        self.records: List[Record] = []

    def fork(self) -> "LocalMemory":
        return copy.deepcopy(self)

    # ---- writes (a write to an existing key replaces it) ----
    def _upsert(self, rec: Record) -> Record:
        for i, r in enumerate(self.records):
            if (r.type, r.key, r.user_id, r.domain) == (rec.type, rec.key, rec.user_id, rec.domain):
                self.records[i] = rec
                return rec
        self.records.append(rec)
        return rec

    def save_semantic(self, text, label, user_id, visibility="private", metadata=None):
        return self._upsert(Record(SEMANTIC, label, text, user_id, visibility, None, dict(metadata or {})))

    def save_taxonomic(self, domain, term, definition, visibility="org", metadata=None):
        return self._upsert(Record(TAXONOMIC, term, definition, None, visibility, domain, dict(metadata or {})))

    def save_procedure(self, procedure, description, content, user_id=None, visibility="org", metadata=None):
        text = "%s %s" % (description, content)
        return self._upsert(Record(PROCEDURAL, procedure, text, user_id, visibility, None, dict(metadata or {})))

    def save_episode(self, title, content, user_id, metadata=None):
        rec = Record(EPISODIC, title, content, user_id, "private", None, dict(metadata or {}))
        self.records.append(rec)  # episodes are a log: never replaced
        return rec

    # ---- reads ----
    def _visible(self, rec: Record, user_id: Optional[str]) -> bool:
        return rec.visibility == "org" or (user_id is not None and rec.user_id == user_id)

    def get_taxonomic_term(self, domain, term) -> Optional[Record]:
        for r in self.records:
            if r.type == TAXONOMIC and r.domain == domain and r.key == term:
                return r
        return None

    def get_procedure(self, procedure) -> Optional[Record]:
        for r in self.records:
            if r.type == PROCEDURAL and r.key == procedure:
                return r
        return None

    def list_episodes(self, user_id) -> List[Record]:
        return [r for r in self.records if r.type == EPISODIC and r.user_id == user_id]

    def search(self, mem_type: str, query: str, user_id: Optional[str] = None, top_k: int = 5) -> List[Chunk]:
        q = set(_tokens(query))
        hits: List[Chunk] = []
        for r in self.records:
            if r.type != mem_type or not self._visible(r, user_id):
                continue
            key_toks = set(_tokens(r.metadata.get("match", r.key)))
            body = set(_tokens(r.text)) | key_toks
            overlap = len(q & body)
            if not overlap:
                continue
            score = overlap / math.sqrt(len(body))
            if key_toks and key_toks <= q:
                score += 2.0  # the query names this record's key outright
            scope = "user" if r.visibility == "private" else "shared"
            hits.append(Chunk(r.type, r.key, r.text, round(score, 4), scope, copy.deepcopy(r.metadata)))
        hits.sort(key=lambda c: (-c.score, c.key))
        return hits[:top_k]

    def build_context_from_sources(self, query, sources: List[SourceSpec], user_id=None,
                                   max_tokens: int = 2048, pinned_procedures=()) -> Context:
        """Retrieve per source, rank together, trim to the token budget."""
        chunks: List[Chunk] = []
        for name in pinned_procedures:  # standing rules are loaded by name, not by similarity
            rec = self.get_procedure(name)
            if rec is not None:
                chunks.append(Chunk(PROCEDURAL, rec.key, rec.text, 99.0, "shared", copy.deepcopy(rec.metadata)))
        found: List[Chunk] = []
        for spec in sources:
            found.extend(self.search(spec.source, query, user_id=user_id, top_k=spec.top_k))
        found.sort(key=lambda c: (-c.score, c.type, c.key))
        budget = max_tokens - FORMAT_RESERVE_TOKENS
        used, kept = 0, []
        for c in chunks + found:
            cost = _estimate_tokens(c.text)
            if used + cost > budget:
                continue
            used += cost
            kept.append(c)
        text = "\n".join("[%s/%s] %s: %s" % (c.type, c.scope, c.key, c.text) for c in kept)
        return Context(kept, text, used)


# --------------------------------------------------------------------------
# Harness document and patches
# --------------------------------------------------------------------------

@dataclass
class Harness:
    """Everything about the agent's environment that Ratchet is allowed to change."""
    version: int = 0
    parent: Optional[int] = None
    context_policy: Dict[str, Any] = field(default_factory=dict)
    tools_offered: List[str] = field(default_factory=list)
    project_policies: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass
class Patch:
    title: str
    target: str            # which part of the harness: tool access, budget, rule, ...
    rationale: str         # the evidence that motivated it
    addresses: str         # the failure diagnosis it is meant to fix
    ops: List[Dict[str, Any]]
    author_user: Optional[str] = None

    def signature(self) -> str:
        import json
        return json.dumps(self.ops, sort_keys=True)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass
class State:
    """One complete, evaluable configuration: harness + memory + policy."""
    harness: Harness
    memory: LocalMemory
    policy: PolicyEngine

    def fork(self) -> "State":
        return State(copy.deepcopy(self.harness), self.memory.fork(), copy.deepcopy(self.policy))

    def apply(self, patch: Patch) -> None:
        """Apply a patch. Raises PolicyRejected if any op exceeds the org ceiling."""
        ops = sorted(patch.ops, key=lambda o: 0 if o["op"] == "set_policy" else 1)  # ceiling check first
        for op in ops:
            kind = op["op"]
            if kind == "set_policy":
                self.policy.set_project(op["type"], op["value"])
                self.harness.project_policies[op["type"]] = copy.deepcopy(op["value"])
            elif kind == "offer_tool":
                if op["tool"] not in self.harness.tools_offered:
                    self.harness.tools_offered.append(op["tool"])
            elif kind == "enable_source":
                self.harness.context_policy.setdefault("sources", {})[op["source"]] = {
                    "mode": op.get("mode", "hybrid"), "top_k": op.get("top_k", 5)}
            elif kind == "pin_procedure":
                pinned = self.harness.context_policy.setdefault("pinned_procedures", [])
                if op["procedure"] not in pinned:
                    pinned.append(op["procedure"])
            elif kind == "save_taxonomic":
                self.memory.save_taxonomic(op["domain"], op["term"], op["definition"], metadata=op.get("metadata"))
            elif kind == "save_semantic":
                self.memory.save_semantic(op["text"], op["label"], op["user_id"], metadata=op.get("metadata"))
            elif kind == "save_procedure":
                self.memory.save_procedure(op["procedure"], op["description"], op["content"],
                                           metadata=op.get("metadata"))
            else:
                raise ValueError("unknown patch op: %s" % kind)
