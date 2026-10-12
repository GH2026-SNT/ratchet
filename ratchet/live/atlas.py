"""MongoDB Atlas: store Ratchet's records and run the agent's queries on real collections.

Needs:  pip install pymongo     and     MONGODB_URI set to your Atlas connection string.
A free Atlas cluster is enough for this module.

UNTESTED against a live cluster (no network access where this was written).
"""
from __future__ import annotations

import copy
import datetime as _dt
from typing import Any, Dict, List

from ..world import build_dataset, schema_of


def connect(uri, db_name: str = "ratchet"):
    if not uri:
        raise SystemExit("Set MONGODB_URI to your Atlas connection string first (see README, step 2).")
    try:
        from pymongo import MongoClient
    except ImportError:
        raise SystemExit("pymongo is not installed. Run:  pip install pymongo")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")   # fail early with a clear error if the cluster is unreachable
    return client[db_name]


def seed_dataset(mdb, seed: int = 7) -> None:
    """Load the demo dataset into Atlas (replaces the two demo collections)."""
    for name, docs in build_dataset(seed).items():
        mdb[name].drop()
        mdb[name].insert_many(copy.deepcopy(docs))


class MongoStore:
    """Every harness version and every attempt, with its evidence, as documents in Atlas."""

    def __init__(self, mdb, run_id: str = ""):
        self.mdb = mdb
        self.run_id = run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def save_version(self, version: Dict[str, Any]) -> None:
        doc = dict(copy.deepcopy(version), _id="%s:v%d" % (self.run_id, version["version"]), run_id=self.run_id)
        self.mdb["harness_versions"].replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def save_attempt(self, attempt: Dict[str, Any]) -> None:
        self.mdb["attempts"].insert_one(dict(copy.deepcopy(attempt), run_id=self.run_id))


class AtlasTools:
    """Same tools as LocalTools, backed by Atlas.

    Evaluation must never change real data, so delete_documents is intercepted:
    it counts what WOULD be deleted and records it as a violation without deleting.
    """

    def __init__(self, mdb):
        self.mdb = mdb
        self.mutations = 0
        self.exports = 0

    def get_schema(self, collection: str) -> Dict[str, str]:
        doc = self.mdb[collection].find_one()
        return schema_of([doc] if doc else [])

    def run_query(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return list(self.mdb[collection].aggregate(pipeline))

    def delete_documents(self, collection: str, filter: Dict[str, Any]) -> Dict[str, int]:
        would_delete = self.mdb[collection].count_documents(filter)
        self.mutations += would_delete
        return {"deleted": would_delete}

    def export_csv(self, collection: str) -> Dict[str, int]:
        rows = self.mdb[collection].estimated_document_count()
        self.exports += rows
        return {"rows": rows}
