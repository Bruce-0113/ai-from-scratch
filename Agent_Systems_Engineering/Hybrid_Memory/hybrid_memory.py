"""Mem0-shaped hybrid memory: vector + KV + graph with fusion scoring.

Three stores run side by side, each tuned for a different query shape
(semantic recall, exact fact lookup, relationship reasoning), and a fusion
layer merges their hits into one ranked list on `Mem0.search()`. Stdlib
only: the vector store's "embedding" is token-overlap (Jaccard) similarity,
a stand-in for a real embedding model. Scope taxonomy: user (persists
across sessions) / session (persists within one thread) / agent (shared
across every caller). Fusion score = relevance + importance + recency,
weights tunable per product via `Mem0Config`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Record:
    """One memory write, shared by all three stores.

    `scope` drives the isolation rules in `Mem0._visible`: "user" records
    are private to `user_id`; "session" records are private to
    `(user_id, session_id)`; "agent" records are visible to any caller.
    """

    rid: str
    text: str
    scope: str
    user_id: str
    session_id: str
    importance: float = 0.5
    ts: float = field(default_factory=time.time)
    tags: tuple[str, ...] = ()


class VectorStore:
    """Semantic recall over free text.

    `search` stands in for an embedding model with plain token-overlap
    (Jaccard) similarity, so the whole demo stays dependency-free and
    deterministic.
    """

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def add(self, record: Record) -> None:
        self._records[record.rid] = record

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Record]]:
        """Return the `top_k` records with the highest token-overlap score.

        Records with zero overlap are dropped rather than ranked last.
        """
        q_tokens = set(query.lower().split())
        scored: list[tuple[float, Record]] = []
        for record in self._records.values():
            r_tokens = set(record.text.lower().split())
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens)
            if overlap == 0:
                continue
            score = overlap / (len(q_tokens | r_tokens))
            scored.append((score, record))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


@dataclass(frozen=True)
class KVKey:
    """Exact-match address for a fact: `(user_id, fact_type, entity)`.

    `entity` holds the fact's current value (e.g. `"Lisbon"`), not a name —
    so writing a new value for an existing `fact_type` (a city change)
    produces a *new* key instead of overwriting the old one. The stale fact
    stays reachable through `KVStore.by_user`; see the README for why that
    is a real limitation of this store rather than intended versioning.
    """

    user_id: str
    fact_type: str
    entity: str


class KVStore:
    """Exact fact lookup, keyed on `(user_id, fact_type, entity)` for O(1)
    point queries such as "what is the user's phone number"."""

    def __init__(self) -> None:
        self._map: dict[KVKey, Record] = {}

    def put(self, key: KVKey, record: Record) -> None:
        self._map[key] = record

    def get(self, key: KVKey) -> Record | None:
        return self._map.get(key)

    def by_user(self, user_id: str) -> list[Record]:
        return [r for k, r in self._map.items() if k.user_id == user_id]


@dataclass
class Edge:
    subject: str
    relation: str
    obj: str
    valid: bool = True
    ts: float = field(default_factory=time.time)


class GraphStore:
    """Typed relationship edges for multi-hop and temporal reasoning (Mem0g).

    Conflicting facts are handled non-destructively: adding a new edge for
    a `(subject, relation)` pair that already has a valid edge marks the
    old one invalid instead of deleting it, so "where did ava used to
    live" stays answerable via `neighbors(valid_only=False)`.
    """

    def __init__(self) -> None:
        self._edges: list[Edge] = []

    def add_edge(self, subject: str, relation: str, obj: str) -> None:
        for edge in self._edges:
            if edge.valid and edge.subject == subject and edge.relation == relation:
                edge.valid = False
        self._edges.append(Edge(subject=subject, relation=relation, obj=obj))

    def neighbors(self, subject: str, valid_only: bool = True) -> list[Edge]:
        return [e for e in self._edges
                if e.subject == subject and (e.valid or not valid_only)]

    def all_edges(self) -> list[Edge]:
        return list(self._edges)


@dataclass
class Mem0Config:
    """Fusion weights. Not normalized to sum to 1 — tune per product (a
    chat agent weights recency higher, a compliance agent weights
    importance higher)."""

    w_relevance: float = 0.6
    w_importance: float = 0.2
    w_recency: float = 0.2
    recency_halflife_s: float = 86400.0


class Mem0:
    """Facade over the three stores: `add()` fans a write out to whichever
    of them are relevant, `search()` fuses their hits into one
    relevance-ranked list."""

    # KV hits match on exact fact identity, not text overlap, so they have
    # no real relevance score. This flat placeholder is high enough that a
    # fact can still surface, low enough that it won't out-rank a strong
    # vector match.
    _KV_PSEUDO_RELEVANCE = 0.4

    def __init__(self, config: Mem0Config | None = None) -> None:
        self.vector = VectorStore()
        self.kv = KVStore()
        self.graph = GraphStore()
        self.config = config or Mem0Config()
        self._counter = 0

    def add(self, text: str, *, user_id: str, session_id: str = "s0",
            scope: str = "user", importance: float = 0.5,
            tags: tuple[str, ...] = (),
            kv_triples: tuple[tuple[str, str], ...] = (),
            graph_triples: tuple[tuple[str, str, str], ...] = ()) -> str:
        """Write one memory to every store it's relevant to: always to the
        vector store, plus the KV store for each `(fact_type, entity)` pair
        in `kv_triples` and the graph store for each `(subject, relation,
        object)` triple in `graph_triples`. Returns the new record id.
        """
        self._counter += 1
        rid = f"m{self._counter:03d}"
        record = Record(rid=rid, text=text, scope=scope, user_id=user_id,
                        session_id=session_id, importance=importance, tags=tags)
        self.vector.add(record)
        for fact_type, entity in kv_triples:
            self.kv.put(KVKey(user_id=user_id, fact_type=fact_type, entity=entity), record)
        for subject, relation, obj in graph_triples:
            self.graph.add_edge(subject, relation, obj)
        return rid

    def _recency_score(self, record: Record, now: float) -> float:
        """Exponential decay: 1.0 at write time, 0.5 after one half-life."""
        elapsed = max(0.0, now - record.ts)
        half = self.config.recency_halflife_s
        return 0.5 ** (elapsed / half) if half > 0 else 1.0

    def _visible(self, record: Record, *, user_id: str,
                 scope: str | None, session_id: str | None) -> bool:
        """Scope-taxonomy gate, applied identically to vector and KV hits
        before fusion so isolation can't drift between the two paths.

        "user" records are private to `user_id`; "session" records are
        private to `user_id` and, if `session_id` is given, to that
        session too; "agent" records are visible to any caller.
        """
        if scope is not None and record.scope != scope:
            return False
        if record.scope == "user" and record.user_id != user_id:
            return False
        if record.scope == "session":
            if record.user_id != user_id:
                return False
            if session_id is not None and record.session_id != session_id:
                return False
        return True

    def search(self, query: str, *, user_id: str, scope: str | None = None,
               session_id: str | None = None, top_k: int = 5) -> list[tuple[float, Record]]:
        """Fuse vector and KV recall into one ranked list.

        Both paths score `relevance + importance + recency` with the
        weights in `self.config` and pass through the same `_visible`
        gate; they differ only in where `relevance` comes from (real token
        overlap for vector hits, `_KV_PSEUDO_RELEVANCE` for KV hits). A
        record already surfaced by the vector path is not re-scored by the
        KV path.
        """
        now = time.time()
        vector_hits = self.vector.search(query, top_k=top_k * 3)
        fused: dict[str, tuple[float, Record]] = {}
        for rel, record in vector_hits:
            if not self._visible(record, user_id=user_id, scope=scope, session_id=session_id):
                continue
            recency = self._recency_score(record, now)
            score = (self.config.w_relevance * rel
                     + self.config.w_importance * record.importance
                     + self.config.w_recency * recency)
            fused[record.rid] = (score, record)
        for record in self.kv.by_user(user_id):
            if record.rid in fused:
                continue
            if not self._visible(record, user_id=user_id, scope=scope, session_id=session_id):
                continue
            recency = self._recency_score(record, now)
            score = (self.config.w_relevance * self._KV_PSEUDO_RELEVANCE
                     + self.config.w_importance * record.importance
                     + self.config.w_recency * recency)
            fused[record.rid] = (score, record)
        ordered = sorted(fused.values(), key=lambda x: -x[0])
        return ordered[:top_k]


def main() -> None:
    print("=" * 70)
    print("MEM0 HYBRID MEMORY — Phase 14, Lesson 09")
    print("=" * 70)

    mem = Mem0()

    mem.add(
        "ava prefers citation-heavy, terse writing over tutorial style",
        user_id="ava", session_id="s001",
        importance=0.7, tags=("preference", "writing"),
        kv_triples=(("writing_style", "terse_citation_heavy"),),
    )
    mem.add(
        "ava is building a 30-lesson curriculum on agent engineering",
        user_id="ava", session_id="s001",
        importance=0.9, tags=("project",),
        kv_triples=(("project", "agent_curriculum"),),
        graph_triples=(("ava", "owns_project", "agent_curriculum"),),
    )
    mem.add(
        "ava lives in Berlin",
        user_id="ava", session_id="s001",
        importance=0.6, tags=("profile",),
        kv_triples=(("city", "Berlin"),),
        graph_triples=(("ava", "lives_in", "Berlin"),),
    )
    mem.add(
        "ava moved to Lisbon last month",
        user_id="ava", session_id="s002",
        importance=0.8, tags=("profile", "update"),
        kv_triples=(("city", "Lisbon"),),
        graph_triples=(("ava", "lives_in", "Lisbon"),),
    )
    mem.add(
        "reminder: send the Lisbon relocation paperwork to HR by Friday",
        user_id="ava", session_id="s002", scope="session",
        importance=0.4, tags=("reminder",),
    )
    mem.add(
        "bob requested a refund for invoice 4711",
        user_id="bob", session_id="s010",
        importance=0.9, tags=("billing",),
        kv_triples=(("refund_request", "4711"),),
    )

    print("\nvector-only recall for 'writing style preferences'")
    for score, record in mem.vector.search("writing style preferences", top_k=3):
        print(f"  {score:.3f}  {record.rid}  {record.text}")

    print("\ngraph recall for entities linked to 'ava'")
    for edge in mem.graph.neighbors("ava", valid_only=False):
        status = "VALID  " if edge.valid else "INVALID"
        print(f"  [{status}] {edge.subject} --{edge.relation}--> {edge.obj}")

    print("\nKV point lookup: ava's current city, no ranking needed")
    hit = mem.kv.get(KVKey(user_id="ava", fact_type="city", entity="Lisbon"))
    print(f"  {hit.rid}  {hit.text}" if hit else "  no match")

    print("\nKV recall for ava, all facts")
    print("  (Berlin AND Lisbon both show up: KVStore has no overwrite/")
    print("   versioning, unlike GraphStore's invalidation above — see README)")
    for record in mem.kv.by_user("ava"):
        print(f"  {record.rid}  {record.text}")

    print("\nfused top-3 for ava, query 'where does ava live'")
    for score, record in mem.search("where does ava live", user_id="ava", top_k=3):
        print(f"  {score:.3f}  {record.rid}  {record.text}")

    print("\nfused top-3 for ava, query 'what is she building'")
    for score, record in mem.search("what is ava building", user_id="ava", top_k=3):
        print(f"  {score:.3f}  {record.rid}  {record.text}")

    print("\nuser-scope isolation: bob's refund does not leak to ava's search")
    hits = mem.search("refund invoice", user_id="ava", top_k=5)
    print(f"  ava results: {len(hits)}  (expect 0 user-scoped hits from bob)")
    for score, record in hits:
        print(f"    {score:.3f}  {record.user_id}  {record.text}")

    print("\nsession-scope isolation: the HR reminder only surfaces in its own session")
    same_session = mem.search("relocation paperwork reminder", user_id="ava",
                               scope="session", session_id="s002", top_k=3)
    other_session = mem.search("relocation paperwork reminder", user_id="ava",
                                scope="session", session_id="s001", top_k=3)
    print(f"  session s002 (where it was written): {len(same_session)} hit(s)")
    print(f"  session s001 (a different thread):   {len(other_session)} hit(s) (expect 0)")

    print()
    print("fusion: relevance + importance + recency. per-product weight tuning.")


if __name__ == "__main__":
    main()
