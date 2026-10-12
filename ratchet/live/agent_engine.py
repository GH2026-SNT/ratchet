"""Atlas Agent Engine: long-term memory and policy reads.

STATUS: UNTESTED. Written from the public docs for agent-engine-sdk-memory 0.11.8
and the Policy Engine page (public preview, October 2026). The preview API can
change. Check these assumptions on first run:

  1. save_* methods accept a `metadata=` dict (the SDK README says the dict you
     write comes back under chunk.metadata["metadata"]; the keyword name is assumed).
  2. MemoryChunk exposes its text as `.content` (falls back to `.text`).
  3. search_* methods accept `top_k=`.
  4. Shared records are read with an UNBOUND handle; a user's own records with a
     handle bound to that user (documented: the two reads do not combine in one call).

Setup (see README, step 4):
    pip install agent-engine-sdk-memory
    export AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN=...   AGENTIC_MEMORY_PROJECT_ID=...
"""
from __future__ import annotations

import base64
import copy
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ..core import (EPISODIC, PROCEDURAL, SEMANTIC, TAXONOMIC, Chunk, Context, LocalMemory, Record,
                    SourceSpec, _estimate_tokens, FORMAT_RESERVE_TOKENS)

WRITER_ID = "ratchet-editor"   # identity used for shared writes (the SDK requires a user_id on writes)


class AgentEngineMemory:
    """LocalMemory's interface on top of the Agent Engine memory service."""

    def __init__(self, memory=None):
        if memory is None:
            from agent_engine_sdk_memory import Memory
            memory = Memory()   # reads AGENTIC_MEMORY_* from the environment
        self.mem = memory

    def _bound(self, user_id: str, session_id: str = "ratchet"):
        from agent_engine_sdk_memory import MemoryRequestContext
        return self.mem.bind(MemoryRequestContext(user_id=user_id, session_id=session_id))

    # ---- writes ----
    def save_semantic(self, text, label, user_id, visibility="private", metadata=None):
        md = dict(metadata or {}, rk_type=SEMANTIC, rk_key=label)
        return self._bound(user_id).save_semantic(text=text, label=label, visibility=visibility, metadata=md)

    def save_taxonomic(self, domain, term, definition, visibility="org", metadata=None):
        md = dict(metadata or {}, rk_type=TAXONOMIC, rk_key=term)
        return self._bound(WRITER_ID).save_taxonomic(domain=domain, term=term, definition=definition,
                                                     visibility=visibility, metadata=md)

    def save_procedure(self, procedure, description, content, user_id=None, visibility="org", metadata=None):
        md = dict(metadata or {}, rk_type=PROCEDURAL, rk_key=procedure)
        return self._bound(user_id or WRITER_ID).save_procedure(
            procedure=procedure, description=description, content=content, visibility=visibility, metadata=md)

    def save_episode(self, title, content, user_id, metadata=None):
        md = dict(metadata or {}, rk_type=EPISODIC, rk_key=title)
        return self._bound(user_id).save_episode(title=title, content=content, metadata=md)

    # ---- reads ----
    @staticmethod
    def _chunk(raw, mem_type: str, scope: str) -> Chunk:
        outer = getattr(raw, "metadata", None) or {}
        md = dict(outer.get("metadata", {}) or {})
        text = getattr(raw, "content", None) or getattr(raw, "text", "") or ""
        return Chunk(md.get("rk_type", mem_type), md.get("rk_key", ""), text,
                     float(getattr(raw, "score", 0.0) or 0.0), scope, md)

    def search(self, mem_type: str, query: str, user_id: Optional[str] = None, top_k: int = 5) -> List[Chunk]:
        out: List[Chunk] = []
        if mem_type == TAXONOMIC:
            out += [self._chunk(c, mem_type, "shared") for c in self.mem.search_taxonomic(query, top_k=top_k)]
        elif mem_type == PROCEDURAL:
            out += [self._chunk(c, mem_type, "shared") for c in self.mem.discover_procedures(query)]
        elif mem_type == SEMANTIC and user_id:
            out += [self._chunk(c, mem_type, "user") for c in self._bound(user_id).search_semantic(query, top_k=top_k)]
        elif mem_type == EPISODIC and user_id:
            out += [self._chunk(c, mem_type, "user") for c in self._bound(user_id).search_episodes(query, top_k=top_k)]
        return out[:top_k]

    def get_taxonomic_term(self, domain, term):
        try:
            return self.mem.get_taxonomic_term(domain, term)
        except Exception:
            return None

    def get_procedure(self, procedure):
        try:
            raw = self.mem.get_procedure(procedure)
        except Exception:
            return None
        if raw is None:
            return None
        text = getattr(raw, "content", None) or getattr(raw, "text", "") or str(raw)
        return Record(PROCEDURAL, procedure, text, None, "org")

    def list_episodes(self, user_id) -> List[Record]:
        out = []
        for raw in self._bound(user_id).list_episodes():
            c = self._chunk(raw, EPISODIC, "user")
            out.append(Record(EPISODIC, c.key, c.text, user_id, "private", None, c.metadata))
        return out

    def build_context_from_sources(self, query, sources: List[SourceSpec], user_id=None,
                                   max_tokens: int = 2048, pinned_procedures=()) -> Context:
        chunks: List[Chunk] = []
        for name in pinned_procedures:
            rec = self.get_procedure(name)
            if rec is not None:
                chunks.append(Chunk(PROCEDURAL, rec.key, rec.text, 99.0, "shared", {}))
        found: List[Chunk] = []
        for spec in sources:
            found += self.search(spec.source, query, user_id=user_id, top_k=spec.top_k)
        found.sort(key=lambda c: -c.score)
        budget, used, kept = max_tokens - FORMAT_RESERVE_TOKENS, 0, []
        for c in chunks + found:
            cost = _estimate_tokens(c.text)
            if used + cost <= budget:
                used += cost
                kept.append(c)
        text = "\n".join("[%s/%s] %s: %s" % (c.type, c.scope, c.key, c.text) for c in kept)
        return Context(kept, text, used)


class StagedMemory:
    """Candidate changes are staged locally and reach the real memory only if the gate accepts.

    Reads see the base memory plus the staged writes (staged wins on the same key).
    fork() gives the candidate its own staging area. commit() promotes staged writes.
    The editor's attempt log (episodes) is not part of a candidate, so it is written straight through.
    """

    def __init__(self, base, staged: Optional[LocalMemory] = None, pending: Optional[list] = None):
        self.base = base
        self.staged = staged or LocalMemory()
        self.pending: List[Any] = pending or []

    def fork(self) -> "StagedMemory":
        return StagedMemory(self.base, self.staged.fork(), list(self.pending))

    def commit(self) -> None:
        for method, args, kwargs in self.pending:
            getattr(self.base, method)(*args, **kwargs)
        self.pending = []
        self.staged = LocalMemory()

    def _stage(self, method, *args, **kwargs):
        self.pending.append((method, copy.deepcopy(args), copy.deepcopy(kwargs)))
        return getattr(self.staged, method)(*args, **kwargs)

    def save_semantic(self, *a, **k): return self._stage("save_semantic", *a, **k)
    def save_taxonomic(self, *a, **k): return self._stage("save_taxonomic", *a, **k)
    def save_procedure(self, *a, **k): return self._stage("save_procedure", *a, **k)
    def save_episode(self, *a, **k): return self.base.save_episode(*a, **k)
    def list_episodes(self, user_id): return self.base.list_episodes(user_id)

    def get_taxonomic_term(self, domain, term):
        return self.staged.get_taxonomic_term(domain, term) or self.base.get_taxonomic_term(domain, term)

    def get_procedure(self, procedure):
        return self.staged.get_procedure(procedure) or self.base.get_procedure(procedure)

    @property
    def records(self):
        return list(getattr(self.base, "records", [])) + list(self.staged.records)

    def build_context_from_sources(self, query, sources, user_id=None, max_tokens=2048, pinned_procedures=()):
        new = self.staged.build_context_from_sources(query, sources, user_id, max_tokens, pinned_procedures)
        old = self.base.build_context_from_sources(query, sources, user_id, max_tokens, pinned_procedures)
        seen = {(c.type, c.key, c.scope) for c in new.chunks}
        chunks = new.chunks + [c for c in old.chunks if (c.type, c.key, c.scope) not in seen]
        text = "\n".join("[%s/%s] %s: %s" % (c.type, c.scope, c.key, c.text) for c in chunks)
        return Context(chunks, text, sum(_estimate_tokens(c.text) for c in chunks))


class PolicyAPI:
    """Read-only client for the Agent Engine policy endpoints.

    Documented and used here:
        POST /api/v1/oauth/token                         service account -> access token
        GET  /api/v1/projects/{projectId}/policies/effective
    Writing policies is deliberately NOT automated: the request body schema was not
    confirmed from the docs. Apply an accepted policy change in the platform UI
    (Manage > Policies) or with the documented project policy endpoints.
    """

    BASE = "https://agentengine.mongodb.com"

    def __init__(self, project_id: str, client_id: str, client_secret: str):
        self.project_id = project_id
        basic = base64.b64encode(("%s:%s" % (client_id, client_secret)).encode()).decode()
        req = urllib.request.Request(self.BASE + "/api/v1/oauth/token",
                                     data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
                                     headers={"Authorization": "Basic " + basic})
        with urllib.request.urlopen(req, timeout=20) as r:
            self.token = json.load(r)["access_token"]

    def effective(self) -> Dict[str, Any]:
        req = urllib.request.Request(
            "%s/api/v1/projects/%s/policies/effective" % (self.BASE, self.project_id),
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
