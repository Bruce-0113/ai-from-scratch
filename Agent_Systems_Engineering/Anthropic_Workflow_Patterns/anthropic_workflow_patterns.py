"""All five of Anthropic's workflow patterns, in stdlib only: prompt
chaining, routing, parallelization (voting), orchestrator-workers, and
evaluator-optimizer -- run against a single scripted LLM so every pattern's
trace is deterministic and reproducible without an API key.

Mirrors the "Anthropic's Workflow Patterns: Simple Over Complex" lesson from
rohitg00/ai-engineering-from-scratch
(phases/14-agent-engineering/12-anthropic-workflow-patterns), whose central
claim is Schluntz and Zhang's (Anthropic, Dec 2024) distinction between
*workflows* (predefined code paths; the engineer owns the graph) and
*agents* (the model dynamically directs its own tools and steps; the model
owns the graph). Both have a place -- workflows are cheaper, faster, and
easier to audit; agents unlock open-ended problems at the cost of
harder-to-reason-about failure modes. Five patterns, built on one
"augmented LLM" (a model wired to search/tools/memory), cover most
workflow-shaped problems before an agent is ever justified:

1. prompt chaining -- output of call N is the input to call N+1; use when a
   task has a clean linear decomposition.
2. routing -- a classifier call picks which downstream handler runs; use
   when categorically different inputs need different handling.
3. parallelization (voting shape) -- run the same prompt N times
   concurrently and aggregate by majority; the other shape, sectioning
   (different chunks, one call each), isn't implemented here.
4. orchestrator-workers -- a synthesis step dispatches to whichever workers
   can handle the task and combines their output; looks like an agent loop
   but never loops indefinitely.
5. evaluator-optimizer -- one call proposes, another evaluates; iterate
   until the evaluator passes. Self-Refine, generalized.

Every pattern here is 10-20 lines against a `ScriptedLLM` -- a dict-backed
stand-in for a real API call -- so the whole module runs deterministically
offline; swapping in a real client only means replacing
`ScriptedLLM.__call__` with an actual `client.messages.create(...)`, the
five pattern functions don't change.

1. ScriptedLLM                  -- deterministic stand-in for a real LLM call
2. prompt_chain                 -- pattern 1: sequential calls, each step's
                                    output feeds the next step's input
3. route                        -- pattern 2: classify, then dispatch to a
                                    handler
4. parallel_vote                -- pattern 3 (voting shape): N calls,
                                    majority wins
5. Worker / orchestrator_workers -- pattern 4: dispatch to whichever workers
                                    can handle the task, then synthesize
6. evaluator_optimizer          -- pattern 5: propose, evaluate, refine
                                    until PASS
7. demo_* / main                -- runs all five against one scripted LLM
                                    and prints each trace

Run directly (`python anthropic_workflow_patterns.py`) to reproduce the demo.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable


# --- 1. Scripted LLM: deterministic stand-in for a real API call -----------------------
class ScriptedLLM:
    """Deterministic stand-in for a real LLM API call.

    Every "prompt" is just a dict key. A prompt mapped to a plain string
    always returns that string (the router calls, whose answer never
    changes, use this). A prompt mapped to a list of strings advances one
    step per call and then holds at the last entry -- this is what lets
    `evaluator_optimizer`'s FAIL-then-PASS trace and `parallel_vote`'s mixed
    yes/no votes come from the *same* prompt string called multiple times.

    Every call is recorded in `self.calls`, so callers can report a total
    "LLM calls across all five patterns" figure -- the point of the whole
    module is that this number stays small.
    """

    def __init__(self, script: dict[str, str | list[str]]) -> None:
        self.script = script
        self.index: dict[str, int] = {}
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Look up `prompt` and advance its position if scripted as a list.

        Args:
            prompt: The exact prompt string, used as the dict lookup key.

        Returns:
            The scripted response for this call. A prompt with no script
            entry returns a visible `"[unhandled: ...]"` marker instead of
            raising, so a typo in a demo prompt fails loudly in the printed
            output rather than crashing.
        """
        self.calls.append(prompt)
        value = self.script.get(prompt)
        if isinstance(value, list):
            i = self.index.get(prompt, 0)
            self.index[prompt] = min(i + 1, len(value) - 1)
            return value[i]
        if isinstance(value, str):
            return value
        return f"[unhandled: {prompt}]"


# --- 2. Five workflow patterns -----------------------------------------------------------
def prompt_chain(input_text: str, llm: Callable[[str], str],
                  steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pattern 1 -- run `steps` in sequence, each step's output feeding the next.

    Args:
        input_text: The seed text for the first step.
        llm: Callable standing in for one LLM API call.
        steps: `(label, template)` pairs; `template` is formatted with the
            current text via `template.format(text=current)` before being
            sent to `llm`.

    Returns:
        A trace: one `(label, output)` pair per step, in order.
    """
    current = input_text
    trace: list[tuple[str, str]] = []
    for label, template in steps:
        prompt = template.format(text=current)
        output = llm(prompt)
        trace.append((label, output))
        current = output
    return trace


def route(input_text: str, classifier: Callable[[str], str],
          handlers: dict[str, Callable[[str], str]]) -> tuple[str, str]:
    """Pattern 2 -- classify `input_text`, then dispatch to its handler.

    Args:
        input_text: The text to classify and handle.
        classifier: Callable returning a label for `input_text`.
        handlers: Label -> handler mapping; falls back to
            `handlers["default"]` when the classifier's label has no
            dedicated handler.

    Returns:
        A `(label, output)` pair. If neither the label nor `"default"` has
        a handler, `output` is a visible `"no handler for ..."` message
        instead of a raised exception.
    """
    label = classifier(input_text)
    handler = handlers.get(label) or handlers.get("default")
    if handler is None:
        return label, f"no handler for {label}"
    return label, handler(input_text)


def parallel_vote(prompt: str, llm: Callable[[str], str], n: int = 5) -> tuple[str, Counter]:
    """Pattern 3 (voting shape) -- call `llm` on the same `prompt` `n` times and take the majority.

    This is the "voting" half of parallelization; the other half,
    sectioning (splitting the input into independent chunks and running one
    call per chunk), isn't implemented in this module.

    Args:
        prompt: The single prompt sent unchanged on every call.
        llm: Callable standing in for one LLM API call.
        n: Number of independent calls to make.

    Returns:
        A `(winner, counts)` pair: `winner` is the most common response,
        `counts` is the full `Counter` of every response seen, so the
        margin of the vote is visible, not just who won.
    """
    votes = [llm(prompt) for _ in range(n)]
    counts = Counter(votes)
    winner, _ = counts.most_common(1)[0]
    return winner, counts


@dataclass
class Worker:
    """One specialist in the orchestrator-workers pool.

    Attributes:
        name: Label used in the trace and the synthesized output.
        handles: Predicate deciding whether this worker should run on a
            given task -- lets multiple workers opt into the same task
            (e.g. both `security_reviewer` and `style_reviewer` on one
            change) instead of exclusively picking a single winner.
        fn: The worker's own LLM call, run only when `handles` returns True.
    """
    name: str
    handles: Callable[[str], bool]
    fn: Callable[[str], str]


def orchestrator_workers(task: str, workers: list[Worker],
                          synth: Callable[[list[tuple[str, str]]], str]) -> tuple[str, list[tuple[str, str]]]:
    """Pattern 4 -- run every worker whose `handles(task)` is True, then synthesize.

    Unlike `route`, this isn't exclusive dispatch: any number of workers can
    opt into the same task (see `demo_orchestrator`, where the security
    reviewer always runs alongside whichever specialist reviewers match).
    Unlike a full agent loop, the set of workers to run is decided once, up
    front -- there's no re-planning after a worker's output comes back.

    Args:
        task: The task text passed to every worker's `handles` and `fn`.
        workers: The pool of candidate workers to check against `task`.
        synth: Combines every worker's `(name, output)` pair into one final
            result.

    Returns:
        A `(final, outputs)` pair: `final` is `synth`'s combined result,
        `outputs` is the raw `(name, output)` trace of every worker that ran.
    """
    outputs: list[tuple[str, str]] = []
    for worker in workers:
        if worker.handles(task):
            outputs.append((worker.name, worker.fn(task)))
    return synth(outputs), outputs


def evaluator_optimizer(task: str, proposer: Callable[[str, str | None], str],
                         evaluator: Callable[[str, str], tuple[bool, str]],
                         max_iter: int = 5) -> tuple[str, list[tuple[str, str, str]]]:
    """Pattern 5 -- propose, evaluate, and retry with feedback until the evaluator passes.

    Generalizes Self-Refine: `proposer` sees the previous evaluator's
    feedback (or `None`, on the first attempt) and can use it to produce a
    better candidate on the next round.

    Args:
        task: The task description passed to `proposer` on every attempt.
        proposer: Produces a candidate given the task and the prior
            evaluator feedback (`None` on the first call).
        evaluator: Judges a candidate; returns `(passed, feedback)`.
        max_iter: Maximum number of propose/evaluate rounds before giving up.

    Returns:
        A `(final, trace)` pair. `final` is the last candidate produced --
        the passing one if the evaluator ever approved, otherwise whatever
        the last attempt was. `trace` has one
        `(candidate, "PASS"|"FAIL", feedback)` entry per round.
    """
    trace: list[tuple[str, str, str]] = []
    feedback: str | None = None
    for i in range(max_iter):
        candidate = proposer(task, feedback)
        ok, judge = evaluator(task, candidate)
        trace.append((candidate, "PASS" if ok else "FAIL", judge))
        if ok:
            return candidate, trace
        feedback = judge
    return candidate, trace


# --- 3. Demo -----------------------------------------------------------------------------
def demo_chain(llm: ScriptedLLM) -> None:
    """Demo pattern 1: summarize the input, then title the summary."""
    print("-" * 70)
    print("1. PROMPT CHAINING — summarize then title")
    print("-" * 70)
    trace = prompt_chain(
        input_text="Agents are ReAct loops with tools, memory, and guardrails.",
        llm=llm,
        steps=[
            ("summarize", "summarize: {text}"),
            ("title", "give a 6-word title: {text}"),
        ],
    )
    for label, output in trace:
        print(f"  [{label}] {output}")


def demo_route(llm: ScriptedLLM) -> None:
    """Demo pattern 2: classify three support messages, dispatch each to its handler."""
    print("\n" + "-" * 70)
    print("2. ROUTING — classify then dispatch")
    print("-" * 70)

    def classifier(text: str) -> str:
        return llm(f"classify: {text}")

    handlers = {
        "refund": lambda t: llm(f"handle refund: {t}"),
        "bug": lambda t: llm(f"handle bug: {t}"),
        "sales": lambda t: llm(f"handle sales: {t}"),
        "default": lambda t: "escalate to human",
    }

    for inp in ("I want my money back",
                "the CLI crashes on ctrl-c",
                "do you offer volume pricing"):
        label, out = route(inp, classifier, handlers)
        print(f"  [{label}] {out}")


def demo_parallel(llm: ScriptedLLM) -> None:
    """Demo pattern 3: 5 votes on a yes/no safety question, majority wins."""
    print("\n" + "-" * 70)
    print("3. PARALLELIZATION — N voters on a boolean")
    print("-" * 70)
    winner, counts = parallel_vote("is this code safe to ship?", llm, n=5)
    print(f"  winner: {winner}")
    print(f"  counts: {dict(counts)}")


def demo_orchestrator(llm: ScriptedLLM) -> None:
    """Demo pattern 4: three reviewers; only the matching specialists plus the always-on security reviewer run."""
    print("\n" + "-" * 70)
    print("4. ORCHESTRATOR-WORKERS — specialist pool")
    print("-" * 70)

    workers = [
        Worker("python_reviewer",
               handles=lambda t: "python" in t.lower(),
               fn=lambda t: llm(f"review python: {t}")),
        Worker("security_reviewer",
               handles=lambda t: True,
               fn=lambda t: llm(f"review security: {t}")),
        Worker("style_reviewer",
               handles=lambda t: "style" in t.lower(),
               fn=lambda t: llm(f"review style: {t}")),
    ]

    def synth(outputs: list[tuple[str, str]]) -> str:
        return " | ".join(f"{name}: {out}" for name, out in outputs)

    task = "review this python change for style and security"
    final, outputs = orchestrator_workers(task, workers, synth)
    for name, out in outputs:
        print(f"  [{name}] {out}")
    print(f"  synth: {final}")


def demo_evaluator_optimizer(llm: ScriptedLLM) -> None:
    """Demo pattern 5: a FAIL-then-PASS trace refining a one-line ReAct summary."""
    print("\n" + "-" * 70)
    print("5. EVALUATOR-OPTIMIZER — propose, judge, refine")
    print("-" * 70)

    def proposer(task: str, feedback: str | None) -> str:
        prompt = f"propose: {task}"
        if feedback:
            prompt += f" (fix: {feedback})"
        return llm(prompt)

    def evaluator(task: str, candidate: str) -> tuple[bool, str]:
        verdict = llm(f"evaluate: {candidate}")
        ok = verdict.startswith("PASS")
        return ok, verdict

    final, trace = evaluator_optimizer(
        "write a one-line summary of ReAct", proposer, evaluator
    )
    for i, (cand, verdict, reason) in enumerate(trace, 1):
        print(f"  iter {i}  [{verdict}] {cand}  // {reason}")
    print(f"  final: {final}")


def main() -> None:
    """Run all five demos against one shared ScriptedLLM and print the total call count."""
    print("=" * 70)
    print("ANTHROPIC WORKFLOW PATTERNS — Phase 14, Lesson 12")
    print("=" * 70)

    llm = ScriptedLLM({
        "summarize: Agents are ReAct loops with tools, memory, and guardrails.":
            "Agents: ReAct + tools + memory + guardrails.",
        "give a 6-word title: Agents: ReAct + tools + memory + guardrails.":
            "Agents as ReAct with Guardrails Built In",

        "classify: I want my money back": "refund",
        "classify: the CLI crashes on ctrl-c": "bug",
        "classify: do you offer volume pricing": "sales",
        "handle refund: I want my money back": "refund filed",
        "handle bug: the CLI crashes on ctrl-c": "bug logged",
        "handle sales: do you offer volume pricing": "quote sent",

        "is this code safe to ship?": ["yes", "yes", "no", "yes", "no"],

        "review python: review this python change for style and security":
            "python ok",
        "review security: review this python change for style and security":
            "security ok",
        "review style: review this python change for style and security":
            "style ok",

        "propose: write a one-line summary of ReAct":
            "ReAct loops thoughts and tool calls.",
        "evaluate: ReAct loops thoughts and tool calls.":
            "FAIL: missing observations",
        "propose: write a one-line summary of ReAct (fix: FAIL: missing observations)":
            "ReAct interleaves thought, action, and observation until done.",
        "evaluate: ReAct interleaves thought, action, and observation until done.":
            "PASS",
    })

    demo_chain(llm)
    demo_route(llm)
    demo_parallel(llm)
    demo_orchestrator(llm)
    demo_evaluator_optimizer(llm)

    print(f"\ntotal llm calls across all five patterns: {len(llm.calls)}")
    print("direct API + small helpers. no framework needed.")


if __name__ == "__main__":
    main()
