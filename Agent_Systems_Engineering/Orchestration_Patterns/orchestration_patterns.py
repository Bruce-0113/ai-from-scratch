"""Four multi-agent orchestration topologies: supervisor-worker, swarm,
hierarchical, and debate -- run against a single shared deterministic
router so the four traces are directly comparable.

Mirrors the "Orchestration Patterns: Supervisor, Swarm, Hierarchical"
lesson from rohitg00/ai-engineering-from-scratch
(phases/14-agent-engineering/28-orchestration-patterns), whose central
claim is Anthropic's: "It's about building the right system for your
needs." Four topologies recur across 2026 frameworks (LangGraph, CrewAI,
OpenAI Agents SDK) once a single agent plus workflow patterns stops being
enough:

1. supervisor-worker -- one central router dispatches to specialists;
   specialists never talk to each other, only to the supervisor.
2. swarm / peer-to-peer -- agents hand off directly to one another; no
   central router, fewer hops, but harder to reason about and prone to
   bouncing handoffs (A -> B -> A -> B).
3. hierarchical -- supervisors of supervisors, as nested subgraphs; needed
   once a single supervisor's context budget can't hold every specialist's
   description.
4. debate -- parallel proposers cross-check each other and converge on a
   majority vote; closer to verification than routing.

This module routes the same three-intent customer-service task (refund /
bug / sales) through all four topologies against a shared, deterministic
`classify` function standing in for an LLM router call, and counts `ops`
(one per simulated agent turn) as a stand-in for the real cost metric --
LLM calls -- so the four topologies' relative cost is comparable without
calling an actual model:

1. classify / SPECIALISTS  -- the shared router and terminal handlers
                               every topology routes through
2. supervisor_worker        -- one router, one specialist turn
3. swarm                    -- peer-to-peer handoffs, hop-limited
4. hierarchical             -- two nested routing layers, then a specialist
5. debate                   -- three parallel proposers, majority vote
6. main                     -- runs all four against the same tasks and
                                prints each trace plus its op count

Run directly (`python orchestration_patterns.py`) to reproduce the demo.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable


# --- 1. Shared building blocks: the router and the specialists -----------------------
def classify(text: str) -> str:
    """Classify free-text `text` into "refund", "bug", or "sales" by keyword.

    Stands in for an LLM router call: a real supervisor/swarm/hierarchical
    agent would prompt a model to pick a route, but this module is about
    comparing the four *topologies*, so the router itself is a
    deterministic keyword match shared by all four -- the same task always
    lands on the same intent no matter which pattern examines it, keeping
    the traces comparable side by side.

    "sales" is the catch-all: anything that isn't clearly a refund or a
    bug falls through to it, including pricing/quote wording.
    """
    t = text.lower()
    if "refund" in t:
        return "refund"
    if "crash" in t or "error" in t or "bug" in t:
        return "bug"
    return "sales"


SPECIALISTS: dict[str, Callable[[str], str]] = {
    "refund": lambda t: f"refund handled: {t[:30]}",
    "bug":    lambda t: f"bug logged: {t[:30]}",
    "sales":  lambda t: f"quote sent: {t[:30]}",
}
"""Terminal handler per intent -- what a specialist "does" once a topology
has routed a task to it. Each just echoes back a canned action string; the
point of the demo is the routing shape, not the specialist logic."""


# --- 2. Four topologies over the same task --------------------------------------------
def supervisor_worker(tasks: list[str]) -> tuple[list[str], int]:
    """Route every task through one central supervisor to a specialist.

    Models the "supervisor-worker" topology (LangGraph `create_supervisor`,
    Anthropic orchestrator-workers, CrewAI Hierarchical Process): a single
    routing agent decides which specialist handles a task; specialists
    never talk to each other or to anything but the supervisor. Two ops
    per task -- one routing decision, one specialist turn -- makes this
    the cheapest topology here short of swarm's best case.

    Args:
        tasks: Free-text task descriptions to route and handle.

    Returns:
        A `(trace, ops)` pair: `trace` is one line per step describing
        what happened, `ops` is the total simulated agent turns across all
        tasks -- a stand-in for LLM call count / cost.
    """
    trace: list[str] = []
    ops = 0
    for task in tasks:
        ops += 1
        label = classify(task)
        trace.append(f"supervisor -> {label}")
        specialist = SPECIALISTS[label]
        ops += 1
        trace.append(f"  {label}: {specialist(task)}")
    return trace, ops


def swarm(tasks: list[str]) -> tuple[list[str], int]:
    """Hand each task directly between peer agents with no central router.

    Models the "swarm / peer-to-peer" topology (LangGraph swarm topology,
    OpenAI Agents SDK handoffs): agents pass a task directly to whichever
    peer they think should own it instead of going through a router. Each
    task starts at the first-registered specialist (`"refund"`); on its
    turn, the current agent classifies the task itself, and either handles
    it (its own classification matches where it already is) or hands off
    to the agent it thinks should own it. Capped at 3 handoffs to guard
    against two agents bouncing a task back and forth forever
    (A -> B -> A -> B) -- with a deterministic router this always
    converges within 1-2 hops, so the cap is never actually hit.

    Args:
        tasks: Free-text task descriptions to route and handle.

    Returns:
        A `(trace, ops)` pair -- see `supervisor_worker`.
    """
    trace: list[str] = []
    ops = 0
    for task in tasks:
        current = list(SPECIALISTS)[0]
        hops = 0
        while hops < 3:
            ops += 1
            label = classify(task)
            if current == label:
                trace.append(f"swarm[{current}]: {SPECIALISTS[current](task)}")
                break
            trace.append(f"swarm[{current}] handoff -> {label}")
            current = label
            hops += 1
    return trace, ops


def hierarchical(tasks: list[str]) -> tuple[list[str], int]:
    """Route each task through two nested supervision layers, then a specialist.

    Models the "hierarchical" topology (nested subgraphs in LangGraph,
    nested crews in CrewAI): a top-level supervisor first picks a broad
    department (`customer_ops` for refund/bug, `commercial` for sales),
    then that department's own sub-supervisor picks the specific
    specialist. Real systems reach for this once a single supervisor's
    context budget can't hold every specialist's description; here it's
    simulated as two classification calls against the same task, since
    the point is the extra routing *hop*, not a genuinely different
    decision at each layer. Three ops per task -- top-level route,
    sub-route, specialist turn -- makes this the deepest of the four
    topologies, though not the most expensive.

    Args:
        tasks: Free-text task descriptions to route and handle.

    Returns:
        A `(trace, ops)` pair -- see `supervisor_worker`.
    """
    trace: list[str] = []
    ops = 0
    for task in tasks:
        ops += 1
        top_label = "customer_ops" if classify(task) != "sales" else "commercial"
        trace.append(f"top -> {top_label}")
        ops += 1
        sub_label = classify(task)
        trace.append(f"  {top_label} -> {sub_label}")
        specialist = SPECIALISTS[sub_label]
        ops += 1
        trace.append(f"    {sub_label}: {specialist(task)}")
    return trace, ops


def debate(tasks: list[str]) -> tuple[list[str], int]:
    """Run three parallel proposers and converge on their majority vote.

    Models the "debate" topology (parallel proposers + cross-critique --
    closer to verification than routing): three independently-named
    debaters each classify the task and propose an intent, then the
    majority proposal wins and its specialist handles the task. Five ops
    per task -- three proposals, one convergence (vote-counting) turn, one
    specialist turn -- makes this the most expensive topology here.
    Because all three debaters call the same deterministic `classify`,
    they always agree unanimously in this demo; a real debate draws on
    independent model samples, so proposals can genuinely disagree and
    `Counter.most_common` is doing real work resolving that disagreement.

    Args:
        tasks: Free-text task descriptions to route and handle.

    Returns:
        A `(trace, ops)` pair -- see `supervisor_worker`.
    """
    trace: list[str] = []
    ops = 0
    for task in tasks:
        proposals: list[str] = []
        for debater in ("alpha", "beta", "gamma"):
            ops += 1
            label = classify(task)
            proposals.append(label)
            trace.append(f"{debater} proposes {label}")
        ops += 1
        convergent = Counter(proposals).most_common(1)[0][0]
        specialist = SPECIALISTS[convergent]
        ops += 1
        trace.append(f"debate converges -> {convergent}: {specialist(task)}")
    return trace, ops


# --- 3. Demo ---------------------------------------------------------------------------
def main() -> None:
    """Run the same three tasks through all four topologies and print traces.

    Prints each topology's per-task trace and total `ops`, then the
    takeaway: pick a topology after picking the problem, not before.
    """
    print("=" * 70)
    print("ORCHESTRATION PATTERNS — Phase 14, Lesson 28")
    print("=" * 70)

    tasks = [
        "I need a refund for invoice 4711",
        "the CLI crashes on ctrl-c",
        "do you offer volume pricing?",
    ]

    for name, fn in (
        ("supervisor-worker", supervisor_worker),
        ("swarm",             swarm),
        ("hierarchical",      hierarchical),
        ("debate",            debate),
    ):
        trace, ops = fn(tasks)
        print(f"\n--- {name}  ops={ops} ---")
        for line in trace:
            print(f"  {line}")

    print()
    print("supervisor: cleanest. swarm: shortest. hierarchical: deepest.")
    print("debate: most expensive. pick topology AFTER picking the problem.")


if __name__ == "__main__":
    main()
