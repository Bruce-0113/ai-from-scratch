"""LangGraph-shaped stateful graph in stdlib -- typed state, nodes, conditional
edges, and a checkpoint-every-node runtime that makes durable execution and
human-in-the-loop pausing possible without any external graph engine.

Mirrors the "Stateful Graph Orchestration -- Durable Execution and
Checkpoints" lesson from rohitg00/ai-engineering-from-scratch
(phases/14-agent-engineering/13-langgraph-stateful-graphs), which reframes
an agent as a state machine: state is a typed, first-class value; nodes are
pure functions that read state and return an update; edges (some
conditional) decide what runs next. The one rule that makes the rest of the
lesson work is that a runtime checkpoints state after *every* node --
recovering from a failure at step 38 of 40 means loading that checkpoint and
continuing at step 39, not restarting from step 1. The same checkpoint
machinery is what makes human-in-the-loop possible: state is already
serialized and addressable, so a node can pause the graph, a human can
inspect and edit that state, and the graph resumes from exactly there.

This module builds that core loop with no dependencies -- no LangGraph, no
database, no LLM call:

1. StateGraph              -- typed state, nodes, edges, and conditional
                               edges (registered as {expected router value:
                               destination node})
2. InMemoryCheckpointer     -- append-only history of (node name, state
                               snapshot) per session; a toy stand-in for a
                               real SQLite/Postgres/Redis-backed checkpointer
3. Runner / PausedAtNode    -- runs the graph node by node, checkpointing
                               after each one; a node can request a pause by
                               putting "_pause_reason" in its update, which
                               raises PausedAtNode with the paused state
4. Demo graph               -- a support-ticket triage flow: classify -> one
                               of refund/bug/sales -> a human-approval gate
                               -> send -> END
5. main                     -- runs the demo to the human gate (pauses),
                               inspects checkpoint history, simulates human
                               approval, and resumes to completion

Run directly (`python stateful_graph_orchestration.py`) to reproduce the
demo.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable


# --- 1. State machine primitives: state, nodes, edges --------------------------------
# State flows through every node as a plain dict; Update is what a node
# returns, shallow-merged into State via `{**state, **update}` after it
# runs. NodeFn and Router are the two function shapes a graph is built from.
State = dict[str, Any]
Update = dict[str, Any]
NodeFn = Callable[[State], Update]
Router = Callable[[State], str]

START = "__start__"
END = "__end__"  # sentinel `Runner.run`'s loop checks for to stop walking the graph


@dataclass
class Edge:
    """One transition in the graph: unconditional if `router` is None.

    Attributes:
        src: Source node name.
        dst: Destination node name.
        router: If set, this edge is only taken when `router(state)` is
            truthy. `add_conditional_edges` builds these via `_make_router`;
            `add_edge` leaves this `None`.
    """

    src: str
    dst: str
    router: Router | None = None


class StateGraph:
    """A graph of named nodes and edges, built up before any state exists.

    Building (`add_node`/`add_edge`/`add_conditional_edges`) and running
    (`Runner`) are deliberately separate: the same `StateGraph` can be run
    many times, for many sessions, against a single `Runner`.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, list[Edge]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        """Register a node function under `name`."""
        self.nodes[name] = fn

    def set_entry(self, name: str) -> None:
        """Set the node `Runner.run` starts at when no `resume_from` is given."""
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        """Add an unconditional transition from `src` straight to `dst`."""
        self.edges.setdefault(src, []).append(Edge(src=src, dst=dst))

    def add_conditional_edges(self, src: str, router: Router,
                               targets: dict[str, str]) -> None:
        """Add one branch per `targets` entry, selected by `router(state)`.

        Registers one `Edge` per `(expected_value, dst)` pair in `targets`,
        each firing when `router(state) == expected_value`. Edges are tried
        in `targets` insertion order, so if `router` could match more than
        one expected value, the first one registered wins.

        Args:
            src: Source node name.
            router: Called with the current state after `src` runs; its
                return value is matched against `targets`' keys.
            targets: Maps an expected `router(state)` value to the node
                name to transition to when it matches.
        """
        for value, dst in targets.items():
            self.edges.setdefault(src, []).append(
                Edge(src=src, dst=dst, router=_make_router(router, value))
            )

    def _next(self, current: str, state: State) -> str | None:
        """Return the destination of the first edge out of `current` whose
        router passes (or that has no router), or `None` if none do."""
        for edge in self.edges.get(current, []):
            if edge.router is None or edge.router(state):
                return edge.dst
        return None


def _make_router(router: Router, expected: str) -> Router:
    """Wrap `router` so it reports whether its output equals `expected`."""
    def fn(state: State) -> bool:
        return router(state) == expected
    return fn


# --- 2. Checkpointing: state persisted after every node -------------------------------
class InMemoryCheckpointer:
    """Append-only (node name, state snapshot) history per session.

    A toy stand-in for a real checkpointer (SQLite/Postgres/Redis, in
    LangGraph terms) -- same interface, but history lives only in this
    process's memory and is lost when it exits.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[str, State]]] = {}

    def save(self, session_id: str, step_name: str, state: State) -> None:
        """Append a snapshot of `state` for `session_id` after `step_name` ran.

        Deep-copies `state` so later in-place mutation by the caller (e.g.
        `Runner.run` popping `_pause_reason`) can't retroactively change an
        already-saved snapshot.

        Args:
            session_id: Which run this checkpoint belongs to.
            step_name: Name of the node that just ran (or a synthetic label
                for a manually-recorded step, e.g. a human review).
            state: The state to snapshot.
        """
        self._store.setdefault(session_id, []).append((step_name, copy.deepcopy(state)))

    def load_latest(self, session_id: str) -> tuple[str, State] | None:
        """Return the most recent `(step_name, state)` checkpoint, or `None`."""
        history = self._store.get(session_id, [])
        if not history:
            return None
        return history[-1]

    def history(self, session_id: str) -> list[tuple[str, State]]:
        """Return every checkpoint recorded for `session_id`, oldest first."""
        return list(self._store.get(session_id, []))


# --- 3. Runner: walks the graph, checkpointing after every node -----------------------
class PausedAtNode(Exception):
    """Raised when a node's update requests a pause (human-in-the-loop).

    Attributes:
        node: Name of the node that just ran and requested the pause.
        state: State right after that node ran (already checkpointed),
            with `_pause_reason` removed.
        reason: The value the node put in `_pause_reason`, e.g. "awaiting
            human approval".
    """

    def __init__(self, node: str, state: State, reason: str) -> None:
        super().__init__(reason)
        self.node = node
        self.state = state
        self.reason = reason


class Runner:
    """Executes a `StateGraph` node by node against an `InMemoryCheckpointer`.

    `Runner` only *writes* to its checkpointer as it goes -- it never reads
    from it to resume. Resuming after a pause or a crash is the caller's
    job: read a checkpoint back out (`checkpointer.load_latest`) and pass
    it in as `state_override` alongside `resume_from`.
    """

    def __init__(self, graph: StateGraph, checkpointer: InMemoryCheckpointer) -> None:
        self.graph = graph
        self.checkpointer = checkpointer

    def run(self, session_id: str, initial_state: State,
            resume_from: str | None = None,
            state_override: State | None = None) -> State:
        """Run the graph from `resume_from` (or the entry node) to `END`.

        Args:
            session_id: Identifies this run's checkpoint history.
            initial_state: Starting state for a fresh run. Ignored if
                `state_override` is given.
            resume_from: Node to start at. Defaults to the graph's entry
                node -- pass a later node name to resume mid-graph.
            state_override: If given, used instead of `initial_state` --
                normally a state pulled back out of the checkpointer (plus
                whatever a human edited) when resuming after a pause.

        Returns:
            The state once the graph reaches `END` (or a dead end -- see
            `StateGraph._next`).

        Raises:
            RuntimeError: No entry node is set and `resume_from` is `None`,
                or the walk reaches an unregistered node name.
            PausedAtNode: A node's update included `_pause_reason`.
        """
        state = copy.deepcopy(state_override if state_override is not None else initial_state)
        current = resume_from or self.graph.entry
        if current is None:
            raise RuntimeError("no entry node set")
        while current is not None and current != END:
            fn = self.graph.nodes.get(current)
            if fn is None:
                raise RuntimeError(f"unknown node {current!r}")
            update = fn(state) or {}
            state = {**state, **update}
            self.checkpointer.save(session_id, current, state)
            if state.get("_pause_reason"):
                reason = state.pop("_pause_reason")
                raise PausedAtNode(current, state, reason)
            current = self.graph._next(current, state)
        return state


# --- 4. Demo graph: support-ticket triage with a human-approval gate ------------------
def _classify(state: State) -> Update:
    """Route free-text `input` to refund/bug/sales by keyword match.

    "sales" is the catch-all: anything that matches neither the refund nor
    the bug keywords falls through to it.
    """
    text = state["input"].lower()
    if "refund" in text or "money back" in text:
        route = "refund"
    elif "crash" in text or "bug" in text or "error" in text:
        route = "bug"
    else:
        route = "sales"
    return {"route": route, "step": state.get("step", 0) + 1}


def _refund(state: State) -> Update:
    """Open a refund ticket from the first 12 characters of `input`."""
    return {"ticket": f"REF-{state.get('input', '')[:12]}",
            "step": state.get("step", 0) + 1}


def _bug(state: State) -> Update:
    """Open a bug ticket from the first 12 characters of `input`."""
    return {"ticket": f"BUG-{state.get('input', '')[:12]}",
            "step": state.get("step", 0) + 1}


def _sales(state: State) -> Update:
    """Open a sales ticket from the first 12 characters of `input`."""
    return {"ticket": f"SAL-{state.get('input', '')[:12]}",
            "step": state.get("step", 0) + 1}


def _human_gate(state: State) -> Update:
    """Pause the graph until `state["human_approval"]` is truthy.

    Requests a pause by putting "_pause_reason" in the update -- `Runner.run`
    checks for that key after merging this update into state and raises
    `PausedAtNode` if it's there. This node still counts as having run: its
    `step` increment is applied and checkpointed either way, so resuming
    after approval continues at whatever comes *after* `human_gate`, not at
    `human_gate` itself.
    """
    if not state.get("human_approval"):
        return {"_pause_reason": "awaiting human approval",
                "step": state.get("step", 0) + 1}
    return {"step": state.get("step", 0) + 1}


def _send(state: State) -> Update:
    """Produce the final output string from the ticket opened earlier."""
    return {"output": f"sent {state.get('ticket')}",
            "step": state.get("step", 0) + 1}


def build_graph() -> StateGraph:
    """Build the demo graph: classify -> (refund | bug | sales) -> human_gate -> send -> END."""
    graph = StateGraph()
    graph.add_node("classify", _classify)
    graph.add_node("refund", _refund)
    graph.add_node("bug", _bug)
    graph.add_node("sales", _sales)
    graph.add_node("human_gate", _human_gate)
    graph.add_node("send", _send)
    graph.set_entry("classify")

    # The three route values happen to spell the same names as their
    # destination nodes ("refund" -> node "refund"), but the two are
    # conceptually distinct: the key is the expected router() output, the
    # value is the node to transition to.
    graph.add_conditional_edges(
        "classify",
        router=lambda s: s["route"],
        targets={"refund": "refund", "bug": "bug", "sales": "sales"},
    )
    graph.add_edge("refund", "human_gate")
    graph.add_edge("bug", "human_gate")
    graph.add_edge("sales", "human_gate")
    graph.add_edge("human_gate", "send")
    graph.add_edge("send", END)
    return graph


# --- 5. Demo ---------------------------------------------------------------------------
def main() -> None:
    """Run the demo end to end.

    1. Run a fresh session; it pauses at `human_gate` since
       `human_approval` starts `False` -- catch `PausedAtNode` and print
       where/why it paused.
    2. Print the full checkpoint history recorded up to the pause.
    3. Simulate a human approving: pull the latest checkpoint, flip
       `human_approval` to `True`, clear `_pause_reason`, and record that
       edited state as a new checkpoint entry (for the audit trail -- not
       read back automatically).
    4. Resume the graph from `send` with the approved state passed in as
       `state_override`, and print the final output.
    """
    print("=" * 70)
    print("LANGGRAPH STATE MACHINE — Phase 14, Lesson 13")
    print("=" * 70)

    graph = build_graph()
    ckpt = InMemoryCheckpointer()
    runner = Runner(graph, ckpt)

    session = "s001"
    initial: State = {"input": "the CLI crashes on ctrl-c, please fix",
                       "step": 0, "human_approval": False}

    print("\nfirst run (will pause at human_gate)")
    try:
        final = runner.run(session, initial)
        print(f"  final: {final}")
    except PausedAtNode as paused:
        print(f"  PAUSED at {paused.node}: {paused.reason}")
        print(f"  state at pause: {json.dumps(paused.state, default=str)}")

    print("\ncheckpoint history")
    for node, snap in ckpt.history(session):
        print(f"  {node}  route={snap.get('route')}  "
              f"ticket={snap.get('ticket')}  step={snap.get('step')}")

    print("\nhuman approves; resume from the node after human_gate")
    latest = ckpt.load_latest(session)
    assert latest is not None
    last_node, last_state = latest
    approved_state = {**last_state, "human_approval": True}
    approved_state.pop("_pause_reason", None)
    ckpt.save(session, f"{last_node}_reviewed", approved_state)

    final = runner.run(
        session_id=session,
        initial_state=initial,
        resume_from="send",
        state_override=approved_state,
    )
    print(f"  final: {final}")

    print()
    print("property: state is checkpointed after every node, so resume is exact.")
    print("a real workflow failing at step 38 of 40 resumes at 39, not step 1.")


if __name__ == "__main__":
    main()
