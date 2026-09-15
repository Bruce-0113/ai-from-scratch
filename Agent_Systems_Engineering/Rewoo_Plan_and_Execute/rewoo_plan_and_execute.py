"""Toy ReWOO (Reasoning WithOut Observation) -- Planner, Workers, Solver,
decoupled. Stdlib only.

Mirrors the "ReWOO and Plan-and-Execute: Decoupled Planning" lesson from
rohitg00/ai-engineering-from-scratch (phases/14-agent-engineering/02-rewoo-
plan-and-execute), whose starting point is a critique of ReAct: interleaving
thought/action/observation in one stream means every tool call re-sends the
entire prior trajectory, so prompt length (and cost) grows with step count,
and a failure mid-loop forces the model to re-derive the whole plan from the
error observation. ReWOO (Xu et al., arXiv:2305.18323) bets that planning
does not need to see observations at all: emit the whole plan up front as a
DAG, fetch all the evidence (worker/tool calls -- independent nodes could
run in parallel), then compose the final answer once, in context with the
plan and all the evidence. The paper reports ~5x fewer tokens and +4
accuracy on HotpotQA versus ReAct, and -- because the planner never sees
observations -- a planner trained on a large teacher's traces can be
distilled into a much smaller model without touching the executor.

This module builds the three roles directly:

1. PlanStep / Plan          -- a step names a tool, its arguments, and (via
                                "#E1"-style string references) which earlier
                                steps' outputs it depends on
2. ToolRegistry              -- name -> callable dispatch; the actual
                                "worker" tools a plan can call
3. topological                -- resolves the plan DAG into a dependency-
                                respecting run order, from string
                                references alone, no explicit edge list
4. resolve_references /
   run_workers                -- substitute "#E<n>" placeholders with prior
                                evidence and dispatch each step's tool call
                                in that order
5. ScriptedPlanner /
   ScriptedSolver              -- stand in for the one planner LLM call and
                                the one solver LLM call; scripted here so
                                the demo is deterministic and model-free
6. run_rewoo                  -- runs Planner -> Workers -> Solver end to
                                end and tracks a character count per phase,
                                as a token-use stand-in
7. run_react_mock              -- replays the same tool calls as an
                                interleaved ReAct trajectory (full history
                                resent every step), for comparison
8. main                        -- runs the "population of the capital of
                                France, rounded" demo and prints the
                                ReWOO-vs-ReAct character-count ratio

Run directly (`python rewoo_plan_and_execute.py`) to reproduce the demo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# --- 1. Plan data structures -----------------------------------------------------------
@dataclass
class PlanStep:
    """One node in a plan DAG: call `tool` with `args`, store the result under `id`.

    Attributes:
        id: This step's evidence key (e.g. "E1"), referenced by later steps
            as the string "#E1" inside their own `args`.
        tool: Name of the tool to dispatch through a `ToolRegistry`.
        args: Keyword arguments for the tool call. A string value containing
            "#E<n>" is a reference to an earlier step's output, resolved by
            `resolve_references` just before dispatch.
    """

    id: str
    tool: str
    args: dict[str, Any]


@dataclass
class Plan:
    """An ordered list of `PlanStep`s emitted by a planner for one question.

    List order need not be dependency order -- `topological` computes the
    actual execution order from each step's "#E<n>" references.
    """

    steps: list[PlanStep]


# --- 2. Tool registry --------------------------------------------------------------------
class ToolRegistry:
    """Name -> callable dispatch table for the tools workers can call."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """Register `fn` under `name` so `dispatch` can call it by that name."""
        self._tools[name] = fn

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Call the tool registered as `name` with `args` as keyword arguments.

        Never raises: an unknown tool name or an exception raised inside the
        tool both come back as an `"error: ..."` string, so one failing
        worker step can't crash the whole plan -- the solver sees the error
        text in its evidence just like any other observation.
        """
        fn = self._tools.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            return fn(**args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


# --- 3. Reference resolution & dependency-ordered execution (Workers) -------------------
REFERENCE_RE = re.compile(r"#E(\d+)")  # matches "#E1", "#E2", ... inside a step's args


def resolve_references(value: Any, evidence: dict[str, str]) -> Any:
    """Substitute every "#E<n>" in `value` with `evidence["E<n>"]`.

    Non-string `value`s pass through unchanged. A reference with no matching
    key in `evidence` is left as-is (e.g. `#E9` stays `#E9`) rather than
    raising -- a defensive fallback that `run_workers` never actually
    exercises, since `topological` only lets a step through once every id
    it references is already present in `evidence`.
    """
    if not isinstance(value, str):
        return value
    return REFERENCE_RE.sub(lambda m: evidence.get(f"E{m.group(1)}", m.group(0)),
                            value)


def topological(plan: Plan) -> list[PlanStep]:
    """Order `plan.steps` so every step runs only after the steps it references.

    Repeatedly scans the still-pending steps for one whose "#E<n>"
    references (found by regex over `str(step.args)`, not an explicit edge
    list) are all already-resolved step ids; resolves it, and repeats. This
    is an O(n^2) fixed-point topological sort -- fine for the handful of
    steps a toy plan has, not meant to scale.

    Raises:
        RuntimeError: No pending step could be resolved in a full pass,
            meaning the plan references a step id that doesn't exist, or
            two steps reference each other (a cycle).
    """
    resolved: list[PlanStep] = []
    known: set[str] = set()
    pending = list(plan.steps)
    while pending:
        progress = False
        rest: list[PlanStep] = []
        for step in pending:
            refs = REFERENCE_RE.findall(str(step.args))
            if all(f"E{r}" in known for r in refs):
                resolved.append(step)
                known.add(step.id)
                progress = True
            else:
                rest.append(step)
        if not progress:
            raise RuntimeError("cyclic plan or unresolved reference")
        pending = rest
    return resolved


def run_workers(plan: Plan, tools: ToolRegistry) -> dict[str, str]:
    """Run every step of `plan` in dependency order, returning id -> result.

    For each step (in `topological` order), resolves its "#E<n>" references
    against the evidence collected *so far*, then dispatches the tool call
    and records the result under the step's own id -- so later steps can,
    in turn, reference this one.
    """
    evidence: dict[str, str] = {}
    for step in topological(plan):
        bound_args = {k: resolve_references(v, evidence) for k, v in step.args.items()}
        evidence[step.id] = tools.dispatch(step.tool, bound_args)
    return evidence


# --- 4. Scripted planner & solver (stand-ins for the two LLM calls) ---------------------
class ScriptedPlanner:
    """A fixed `Plan`, handed back regardless of the question asked.

    Stands in for the one real planner LLM call: a real planner reads the
    question and emits a plan DAG; this always returns the same `Plan` it
    was constructed with, which keeps the demo deterministic and model-free.
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def plan_for(self, question: str) -> Plan:
        """Return the scripted plan. `question` is accepted but ignored."""
        return self.plan


class ScriptedSolver:
    """A fixed answer template, filled in with worker evidence.

    Stands in for the one real solver LLM call: a real solver reads the
    question, plan, and evidence and composes an answer in prose; this just
    formats a canned template string against the evidence dict.
    """

    def __init__(self, answer_template: str) -> None:
        self.template = answer_template

    def solve(self, question: str, plan: Plan, evidence: dict[str, str]) -> str:
        """Format the template against `evidence`.

        `question` and `plan` are unused here but kept in the signature
        because a real solver needs both to compose an answer -- only the
        solving logic is scripted, not the interface.
        """
        return self.template.format(**evidence)


# --- 5. Demo tools -----------------------------------------------------------------------
def fake_search(query: str) -> str:
    """Canned lookup table standing in for a real search tool.

    Answers exactly the lookups the demo plan needs (capital of France/
    Germany, population of Paris); anything else returns a visible "no
    result" string instead of raising, mirroring how a real search tool
    degrades on an unanswerable query.
    """
    if "capital of france" in query.lower():
        return "Paris"
    if "population of paris" in query.lower():
        return "11.2 million metro"
    if "capital of germany" in query.lower():
        return "Berlin"
    return f"no result for {query!r}"


def rounded_million(text: str) -> str:
    """Extract the first number in `text` and format it as "N million", rounded.

    E.g. "11.2 million metro" -> "11 million". Returns "unknown" if `text`
    contains no parseable number.
    """
    m = re.search(r"([0-9]+\.?[0-9]*)", text)
    if not m:
        return "unknown"
    return f"{round(float(m.group(1)))} million"


# --- 6. Orchestration: Planner -> Workers -> Solver, with a token-cost stand-in ---------
@dataclass
class ReWOORun:
    """Everything `run_rewoo` produced for one question, including its cost stand-in.

    Attributes:
        question: The question asked.
        plan: The plan the planner returned.
        evidence: Each step's id -> tool result, in execution order.
        answer: The solver's final composed answer.
        planner_chars: Characters "spent" by the one planner call (the
            question plus every step's tool name and args) -- a stand-in
            for planner prompt/output tokens.
        worker_chars: Characters "spent" across all worker calls (each
            step's args plus its result).
        solver_chars: Characters "spent" by the one solver call (the
            question plus all worker output plus the final answer).
    """

    question: str
    plan: Plan
    evidence: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    planner_chars: int = 0
    worker_chars: int = 0
    solver_chars: int = 0


def run_rewoo(question: str, planner: ScriptedPlanner,
              tools: ToolRegistry, solver: ScriptedSolver) -> ReWOORun:
    """Run Planner -> Workers -> Solver once, end to end, for `question`.

    Also approximates each phase's token cost as a character count (real
    tokenization needs a tokenizer; character count is close enough to show
    the *shape* of ReWOO's savings). `worker_chars` pairs each
    `plan.steps` entry positionally with `evidence.values()` -- correct as
    long as a plan is already written in dependency order (true of every
    plan in this demo), but not guaranteed to line up if a plan's step
    order ever diverges from its `topological` execution order.

    Returns:
        A `ReWOORun` with the plan, evidence, final answer, and the three
        character counts.
    """
    plan = planner.plan_for(question)
    planner_chars = len(question) + sum(len(s.tool) + len(str(s.args))
                                        for s in plan.steps)
    evidence = run_workers(plan, tools)
    worker_chars = sum(len(str(s.args)) + len(v) for s, v in zip(plan.steps,
                                                                 evidence.values()))
    answer = solver.solve(question, plan, evidence)
    solver_chars = len(question) + worker_chars + len(answer)
    return ReWOORun(question=question, plan=plan, evidence=evidence,
                    answer=answer,
                    planner_chars=planner_chars, worker_chars=worker_chars,
                    solver_chars=solver_chars)


# --- 7. ReAct comparison ------------------------------------------------------------------
def run_react_mock(question: str, tools: ToolRegistry,
                   trajectory: list[tuple[str, dict[str, Any]]]) -> int:
    """Replay `trajectory` as an interleaved ReAct loop; return its total character cost.

    Unlike `run_rewoo`, every step's simulated prompt re-sends the full
    question plus the running history of every prior action and observation
    (plus a fixed 40-char overhead per step, standing in for the "Thought:"/
    "Action:"/"Observation:" scaffolding a real ReAct prompt repeats) -- this
    is what makes ReAct's cost grow with trajectory length instead of
    staying flat like ReWOO's three fixed-role calls.

    Args:
        question: The question driving this trajectory (resent every step).
        tools: Registry used to actually execute each step's tool call.
        trajectory: The fixed sequence of (tool_name, args) pairs to replay
            -- scripted here to mirror the same plan `run_rewoo` executes,
            so the two totals are comparable.

    Returns:
        Total simulated character cost across the whole trajectory.
    """
    prompt_chars = len(question)
    total = 0
    history_chars = 0
    for name, args in trajectory:
        total += prompt_chars + history_chars + len(name) + len(str(args))
        obs = tools.dispatch(name, args)
        history_chars += len(name) + len(str(args)) + len(obs) + 40
    total += prompt_chars + history_chars
    return total


# --- 8. Demo -----------------------------------------------------------------------------
def main() -> None:
    """Run the "population of the capital of France, rounded" demo.

    Builds a three-step plan (capital -> population -> rounding), runs it
    through `run_rewoo`, and prints the plan, evidence, and final answer.
    Then replays the same three tool calls through `run_react_mock` and
    prints the ReWOO-vs-ReAct character-count ratio as a token-use
    intuition pump (the paper's own claim is ~5x on HotpotQA).
    """
    print("=" * 70)
    print("REWOO — Planner, Workers, Solver (Phase 14, Lesson 02)")
    print("=" * 70)

    tools = ToolRegistry()
    tools.register("search", fake_search)
    tools.register("round_million", rounded_million)

    plan = Plan(steps=[
        PlanStep("E1", "search", {"query": "capital of France"}),
        PlanStep("E2", "search", {"query": "population of #E1"}),
        PlanStep("E3", "round_million", {"text": "#E2"}),
    ])
    planner = ScriptedPlanner(plan)
    solver = ScriptedSolver(
        "The capital of France is {E1}; rounded population is {E3}."
    )
    run = run_rewoo("What is the population of the capital of France, rounded?",
                    planner, tools, solver)

    print("\nPLAN")
    for step in run.plan.steps:
        print(f"  {step.id}: {step.tool}({step.args})")
    print("\nEVIDENCE")
    for k, v in run.evidence.items():
        print(f"  {k} -> {v}")
    print(f"\nFINAL: {run.answer}")

    react_chars = run_react_mock(
        run.question, tools,
        [("search", {"query": "capital of France"}),
         ("search", {"query": "population of Paris"}),
         ("round_million", {"text": "11.2 million metro"})])
    rewoo_chars = run.planner_chars + run.worker_chars + run.solver_chars
    print("\nTOKEN INTUITION (chars, approximate)")
    print(f"  react total  : {react_chars}")
    print(f"  rewoo total  : {rewoo_chars}")
    print(f"  ratio        : {react_chars / max(rewoo_chars, 1):.2f}x")
    print("\npaper claim: ~5x fewer tokens on HotpotQA. toy approximates the shape.")


if __name__ == "__main__":
    main()
