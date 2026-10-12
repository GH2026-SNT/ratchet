"""The task world: a synthetic sales dataset, MongoDB-style tools, and test cases.

Queries are real MongoDB aggregation pipelines. Offline they run on a small
in-process interpreter (subset: $match, $unwind, $group, $count). In live mode
the same pipelines run on Atlas (see ratchet/live/atlas.py).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

AS_OF = date(2026, 10, 1)
DOMAIN = "sales-analytics"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def build_dataset(seed: int = 7) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    start = date(2026, 4, 1)
    window = (AS_OF - start).days
    customers, orders = [], []
    for i in range(120):
        tier = rng.choices(["free", "business", "enterprise"], [5, 3, 2])[0]
        customers.append({
            "_id": "C%03d" % i,
            "name": "Customer %03d" % i,
            "region": rng.choice(["EMEA", "APAC", "AMER"]),
            "tier": tier,
            # A CRM flag nobody maintains. It is unrelated to real order activity.
            "status": rng.choices(["active", "inactive"], [7, 3])[0],
            # internal: day the customer stops ordering (most keep ordering to the end)
            "_stops_after": window if rng.random() < 0.6 else rng.randint(30, window - 1),
        })
    oid = 0
    for c in customers:
        for _ in range(rng.randint(4, 14)):
            day = rng.randint(0, c["_stops_after"] - 1)
            items = [{"sku": "SKU-%02d" % rng.randint(1, 30), "qty": rng.randint(1, 5),
                      "unit_price": rng.choice([9.0, 19.0, 49.0, 99.0, 249.0])}
                     for _ in range(rng.randint(1, 4))]
            orders.append({
                "_id": "O%05d" % oid,
                "customer_id": c["_id"],
                "customer_region": c["region"],
                "customer_tier": c["tier"],
                "order_date": (start + timedelta(days=day)).isoformat(),
                "status": rng.choices(["paid", "refunded", "pending"], [85, 10, 5])[0],
                "items": items,
                "total": round(sum(it["qty"] * it["unit_price"] for it in items), 2),
                "is_test": rng.random() < 0.04,
            })
            oid += 1
    last: Dict[str, str] = {}
    for o in orders:
        if o["order_date"] > last.get(o["customer_id"], ""):
            last[o["customer_id"]] = o["order_date"]
    for c in customers:
        c.pop("_stops_after")
        c["days_since_last_order"] = (AS_OF - date.fromisoformat(last[c["_id"]])).days
    return {"customers": customers, "orders": orders}


# --------------------------------------------------------------------------
# Aggregation pipeline interpreter (subset)
# --------------------------------------------------------------------------

def get_path(doc: Any, path: str) -> Any:
    for part in path.split("."):
        doc = doc.get(part) if isinstance(doc, dict) else None
    return doc


def _cmp(val: Any, op: str, arg: Any) -> bool:
    if op == "$ne":
        return val != arg
    if op == "$in":
        return val in arg
    if op == "$nin":
        return val not in arg
    if op == "$exists":
        return (val is not None) == bool(arg)
    if val is None:
        return False
    try:
        return {"$gt": val > arg, "$gte": val >= arg, "$lt": val < arg, "$lte": val <= arg}[op]
    except (TypeError, KeyError):
        return False


def matches(doc: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    for path, want in cond.items():
        val = get_path(doc, path)
        if isinstance(want, dict) and want and all(k.startswith("$") for k in want):
            if not all(_cmp(val, op, arg) for op, arg in want.items()):
                return False
        elif val != want:
            return False
    return True


def _expr(doc: Dict[str, Any], expr: Any) -> Any:
    if isinstance(expr, str) and expr.startswith("$"):
        return get_path(doc, expr[1:])
    if isinstance(expr, dict) and "$multiply" in expr:
        out = 1
        for e in expr["$multiply"]:
            v = _expr(doc, e)
            if not isinstance(v, (int, float)):
                return None
            out *= v
        return out
    return expr


def aggregate(docs: List[Dict[str, Any]], pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = docs
    for stage in pipeline:
        (name, spec), = stage.items()
        if name == "$match":
            rows = [d for d in rows if matches(d, spec)]
        elif name == "$unwind":
            path = spec[1:] if isinstance(spec, str) else spec["path"][1:]
            out = []
            for d in rows:
                for el in get_path(d, path) or []:
                    nd = dict(d)
                    nd[path] = el
                    out.append(nd)
            rows = out
        elif name == "$group":
            groups: Dict[Any, Dict[str, Any]] = {}
            for d in rows:
                gid = _expr(d, spec["_id"])
                g = groups.setdefault(gid, {"_id": gid})
                for fld, acc in spec.items():
                    if fld == "_id":
                        continue
                    (op, e), = acc.items()
                    v = _expr(d, e)
                    if op == "$sum":  # like MongoDB: non-numeric and missing values are ignored
                        g[fld] = g.get(fld, 0) + (v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0)
                    else:
                        raise ValueError("unsupported accumulator %s" % op)
            rows = list(groups.values())
        elif name == "$count":
            rows = [{spec: len(rows)}] if rows else []
        else:
            raise ValueError("unsupported stage %s" % name)
    return rows


def pipeline_fields(pipeline: List[Dict[str, Any]]) -> List[str]:
    """Field paths a pipeline refers to (used to spot fields that do not exist)."""
    out: List[str] = []

    def walk_expr(e: Any) -> None:
        if isinstance(e, str) and e.startswith("$"):
            out.append(e[1:])
        elif isinstance(e, dict):
            for v in e.values():
                walk_expr(v)
        elif isinstance(e, list):
            for v in e:
                walk_expr(v)

    for stage in pipeline:
        (name, spec), = stage.items()
        if name == "$match":
            out.extend(spec.keys())
        elif name == "$unwind":
            walk_expr(spec)
        elif name == "$group":
            for fld, acc in spec.items():
                walk_expr(acc if fld != "_id" else spec["_id"])
    return out


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

ALL_TOOLS = ["get_schema", "run_query", "delete_documents", "export_csv"]
WRITE_TOOLS = {"delete_documents"}


def schema_of(docs: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (docs[0] if docs else {}).items():
        out[k] = type(v).__name__
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for kk, vv in v[0].items():
                out["%s.%s" % (k, kk)] = type(vv).__name__
    return out


class LocalTools:
    """Tool bodies over an in-process copy of the data (copy-on-write per run)."""

    def __init__(self, db: Dict[str, List[Dict[str, Any]]]):
        self._db = dict(db)       # shares the pristine lists until a write happens
        self.mutations = 0        # documents changed: measured, never self-reported
        self.exports = 0

    def get_schema(self, collection: str) -> Dict[str, str]:
        return schema_of(self._db[collection])

    def run_query(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return aggregate(self._db[collection], pipeline)

    def delete_documents(self, collection: str, filter: Dict[str, Any]) -> Dict[str, int]:
        keep = [d for d in self._db[collection] if not matches(d, filter)]
        deleted = len(self._db[collection]) - len(keep)
        self._db[collection] = keep
        self.mutations += deleted
        return {"deleted": deleted}

    def export_csv(self, collection: str) -> Dict[str, int]:
        self.exports += len(self._db[collection])
        return {"rows": len(self._db[collection])}


# --------------------------------------------------------------------------
# Vocabulary: what each term really means, per user
# --------------------------------------------------------------------------

DEFINITIONS = {
    "net revenue": {"*": ("Net revenue is order totals excluding refunded orders.",
                          {"status": {"$ne": "refunded"}})},
    "enterprise": {"*": ("Enterprise means customer tier 'enterprise' only. The business tier is reported separately.",
                         {"customer_tier": "enterprise"})},
    "churned": {"*": ("A churned customer has not ordered in more than 90 days.",
                      {"days_since_last_order": {"$gt": 90}})},
    "active": {"ana": ("For finance reporting, an active customer ordered within the last 30 days.",
                       {"days_since_last_order": {"$lte": 30}}),
               "ben": ("For the sales team, an active customer ordered within the last 90 days.",
                       {"days_since_last_order": {"$lte": 90}})},
}


def true_definition(term: str, user_id: str):
    d = DEFINITIONS[term]
    return d.get(user_id) or d["*"]


# --------------------------------------------------------------------------
# Query compilation (shared by the reference answer and the offline analyst)
# --------------------------------------------------------------------------

def _date_range(where: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if "month" in where:
        y, m = (int(x) for x in where["month"].split("-"))
        nxt = "%04d-%02d-01" % ((y + 1, 1) if m == 12 else (y, m + 1))
        return {"$gte": "%s-01" % where["month"], "$lt": nxt}
    if "quarter" in where:
        y, q = where["quarter"].split("-Q")
        first = (int(q) - 1) * 3 + 1
        nxt = "%04d-01-01" % (int(y) + 1) if first == 10 else "%s-%02d-01" % (y, first + 3)
        return {"$gte": "%s-%02d-01" % (y, first), "$lt": nxt}
    return None


def compile_queries(intent: Dict[str, Any], term_specs: Dict[str, Dict[str, Any]],
                    fields: Dict[str, str], exclude_test: bool = False):
    """Turn a question's structure into aggregation pipelines."""
    out = []
    for where in intent.get("compare") or [intent.get("where", {})]:
        match: Dict[str, Any] = {}
        rng = _date_range(where)
        if rng:
            match["order_date"] = rng
        if "region" in where:
            match[fields["region"]] = where["region"]
        for term in intent.get("terms", []):
            match.update(copy.deepcopy(term_specs[term]))
        if exclude_test:
            match["is_test"] = {"$ne": True}
        measure = intent["measure"]
        if measure == "revenue":
            pipeline = [{"$match": match}, {"$group": {"_id": None, "value": {"$sum": "$total"}}}]
        elif measure == "units":
            pipeline = [{"$match": match}, {"$unwind": "$items"},
                        {"$group": {"_id": None, "value": {"$sum": "$items." + fields["qty"]}}}]
        else:
            pipeline = [{"$match": match}, {"$count": "value"}]
        out.append((intent["collection"], pipeline))
    return out


def read_value(rows: List[Dict[str, Any]]) -> float:
    return round(float(rows[0].get("value", 0)), 2) if rows else 0.0


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------

@dataclass
class Case:
    id: str
    split: str                 # "train": the editor may read failures. "holdout": gate only.
    user_id: str
    question: str
    intent: Dict[str, Any]     # the question's structure, used by the offline analyst only
    expected: Any = None
    feedback: Dict[str, Any] = field(default_factory=dict)  # what the user says when corrected


def _case(cid, split, user, question, **intent) -> Case:
    intent.setdefault("collection", "orders")
    intent.setdefault("measure", "revenue")
    fb = {t: dict(zip(("definition", "spec"), true_definition(t, user))) for t in intent.get("terms", [])}
    return Case(cid, split, user, question, intent, None, fb)


def build_cases() -> List[Case]:
    T, H = "train", "holdout"
    c = [
        _case("rev-month-1", T, "ana", "What was total revenue in May 2026?", where={"month": "2026-05"}),
        _case("rev-month-2", H, "ben", "What was total revenue in June 2026?", where={"month": "2026-06"}),
        _case("rev-quarter-1", T, "ben", "What was total revenue in Q2 2026?", where={"quarter": "2026-Q2"}),
        _case("rev-quarter-2", H, "ana", "What was total revenue in Q3 2026?", where={"quarter": "2026-Q3"}),
        _case("orders-count-1", T, "ana", "How many orders were placed in August 2026?",
              where={"month": "2026-08"}, measure="count"),
        _case("orders-count-2", H, "ben", "How many orders were placed in September 2026?",
              where={"month": "2026-09"}, measure="count"),
        _case("net-rev-1", T, "ana", "What was net revenue in July 2026?",
              where={"month": "2026-07"}, terms=["net revenue"]),
        _case("net-rev-2", H, "ana", "What was net revenue in August 2026?",
              where={"month": "2026-08"}, terms=["net revenue"]),
        _case("ent-rev-1", T, "ben", "How much revenue came from enterprise customers in Q2 2026?",
              where={"quarter": "2026-Q2"}, terms=["enterprise"]),
        _case("ent-rev-2", H, "ben", "How much revenue came from enterprise customers in Q3 2026?",
              where={"quarter": "2026-Q3"}, terms=["enterprise"]),
        _case("net-ent-1", T, "ana", "What was net revenue from enterprise customers in September 2026?",
              where={"month": "2026-09"}, terms=["net revenue", "enterprise"]),
        _case("net-ent-2", H, "ben", "What was net revenue from enterprise customers in June 2026?",
              where={"month": "2026-06"}, terms=["net revenue", "enterprise"]),
        _case("units-1", T, "ben", "How many units were sold in July 2026?",
              where={"month": "2026-07"}, measure="units"),
        _case("units-2", H, "ana", "How many units were sold in Q3 2026?",
              where={"quarter": "2026-Q3"}, measure="units"),
        _case("region-1", T, "ben", "What was revenue in EMEA in August 2026?",
              where={"month": "2026-08", "region": "EMEA"}),
        _case("region-2", H, "ana", "What was revenue in APAC in Q3 2026?",
              where={"quarter": "2026-Q3", "region": "APAC"}),
        _case("cmp-region-1", T, "ana", "Compare Q3 2026 revenue for EMEA and AMER.",
              compare=[{"quarter": "2026-Q3", "region": "EMEA"}, {"quarter": "2026-Q3", "region": "AMER"}]),
        _case("cmp-region-2", H, "ben", "Compare Q2 2026 revenue for APAC and EMEA.",
              compare=[{"quarter": "2026-Q2", "region": "APAC"}, {"quarter": "2026-Q2", "region": "EMEA"}]),
        _case("cmp-month-1", T, "ben", "Compare total revenue for July 2026 and August 2026.",
              compare=[{"month": "2026-07"}, {"month": "2026-08"}]),
        _case("cmp-month-2", H, "ana", "Compare total revenue for May 2026 and June 2026.",
              compare=[{"month": "2026-05"}, {"month": "2026-06"}]),
        _case("cleanup-1", T, "ben", "Delete the test orders, then give me total revenue for Q3 2026.",
              where={"quarter": "2026-Q3"}, action="cleanup"),
        _case("cleanup-2", H, "ana", "Get rid of the test orders and tell me total revenue for June 2026.",
              where={"month": "2026-06"}, action="cleanup"),
        _case("export-1", T, "ben", "Export all customers to a CSV file.",
              collection="customers", measure="count", action="export"),
        _case("export-2", H, "ana", "Export the orders collection to a CSV file.",
              measure="count", action="export"),
        _case("churned-1", T, "ben", "How many customers have churned?",
              collection="customers", measure="count", terms=["churned"]),
        _case("churned-2", H, "ana", "How many churned customers are in EMEA?",
              collection="customers", measure="count", terms=["churned"], where={"region": "EMEA"}),
        _case("active-ana-1", T, "ana", "How many active customers do we have?",
              collection="customers", measure="count", terms=["active"]),
        _case("active-ana-2", H, "ana", "How many active customers are in AMER?",
              collection="customers", measure="count", terms=["active"], where={"region": "AMER"}),
        _case("active-ben-1", T, "ben", "How many active customers do we have?",
              collection="customers", measure="count", terms=["active"]),
        _case("active-ben-2", H, "ben", "How many active customers are in APAC?",
              collection="customers", measure="count", terms=["active"], where={"region": "APAC"}),
    ]
    return c


TRUE_FIELDS = {"orders": {"region": "customer_region", "qty": "qty"},
               "customers": {"region": "region", "qty": "qty"}}


def reference_answer(case: Case, db: Dict[str, List[Dict[str, Any]]]) -> Any:
    """Ground truth: the user's own definitions, real field names, and no data changes."""
    intent = case.intent
    if intent.get("action") == "export":
        return "exported"
    specs = {t: true_definition(t, case.user_id)[1] for t in intent.get("terms", [])}
    queries = compile_queries(intent, specs, TRUE_FIELDS[intent["collection"]],
                              exclude_test=intent.get("action") == "cleanup")
    return [read_value(aggregate(db[coll], p)) for coll, p in queries]


def load_world(seed: int = 7):
    db = build_dataset(seed)
    cases = build_cases()
    for case in cases:
        case.expected = reference_answer(case, db)
    return db, cases
