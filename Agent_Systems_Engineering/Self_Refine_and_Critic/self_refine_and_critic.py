"""Toy Self-Refine and CRITIC loop. Stdlib only.

Mirrors the "Self-Refine and CRITIC: Iterative Output Improvement" lesson
from rohitg00/ai-engineering-from-scratch (phases/14-agent-engineering/05-
self-refine-and-critic), which contrasts two ways to close a generate ->
critique -> refine loop. Self-Refine (Madaan et al., arXiv:2303.17651) runs
one LLM in three roles -- generate, feedback, refine -- and reports +20
absolute averaged across 7 tasks, with no training and no external tools.
Its catch: LLMs are unreliable at judging their own factual claims. CRITIC
(Gou et al., arXiv:2305.11738) replaces the feedback step with a verifier
grounded in an external tool (search, code interpreter, calculator, tests),
so the critique the refiner sees is backed by something other than the
model's own confidence.

This module runs both loops against the same scripted generator, on a toy
task: produce a 3-bullet summary, each bullet under 60 chars, containing
none of the errors in `KNOWN_WRONG_FACTS`.

1. Attempt          -- one iteration's record: the output, the critique it
                        drew, and whether that critique marked it verified
2. generate          -- stands in for the LLM's "generate" call; scripted
                        to start with two wrong facts (Paris/Germany,
                        Everest/Europe) and correct one fact at a time --
                        but only when the *previous* critique's text
                        literally contains the wrong term ("germany" /
                        "everest"); otherwise it just repeats the last
                        output unchanged
3. feedback_self     -- Self-Refine's self-critique: correctly flags the
                        Germany error in plain English, but its wording
                        never contains the literal word "germany", so it
                        can never unstick `generate`'s keyword match
4. verify_external   -- CRITIC's grounded verifier: checks the output
                        against `KNOWN_WRONG_FACTS`, bullet count, and
                        line length, and phrases its critique ("...
                        contradicts reference data") so that it does
                        contain the literal wrong term
5. refine / run_loop -- `refine` just re-invokes `generate` with the run's
                        history; `run_loop` is the shared generate ->
                        verify -> refine loop, picking `feedback_self` or
                        `verify_external` as the verify step depending on
                        `use_critic`
6. print_run / main  -- runs both loops from the same starting output and
                        prints their traces side by side: Self-Refine
                        repeats the same critique for all 4 iterations and
                        never converges; CRITIC fixes one fact per
                        iteration and passes on iteration 3

Run directly (`python self_refine_and_critic.py`) to reproduce the demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


KNOWN_WRONG_FACTS = [
    "paris is the capital of germany",
    "mt everest is in europe",
    "the sun orbits the earth",
]


@dataclass
class Attempt:
    iteration: int
    output: str
    critique: str
    verified: bool


def generate(topic: str, history: list[Attempt]) -> str:
    if not history:
        return (
            "- Paris is the capital of Germany\n"
            "- Mt Everest is in Europe\n"
            "- Water boils at 100C"
        )
    last = history[-1]
    if "germany" in last.critique.lower():
        return (
            "- Paris is the capital of France\n"
            "- Mt Everest is in Europe\n"
            "- Water boils at 100C"
        )
    if "everest" in last.critique.lower():
        return (
            "- Paris is the capital of France\n"
            "- Mt Everest is in Asia\n"
            "- Water boils at 100C at sea level"
        )
    return history[-1].output


def feedback_self(output: str) -> tuple[str, bool]:
    if "Germany" in output and "Paris" in output:
        return "first bullet reads wrong, double-check capital", False
    if "Europe" in output and "Everest" in output:
        return "second bullet's continent looks off to me", False
    return "no issues", True


def verify_external(output: str) -> tuple[str, bool]:
    text = output.lower()
    for fact in KNOWN_WRONG_FACTS:
        key = fact.split(" is ")[0] if " is " in fact else fact
        if "paris" in text and "germany" in text:
            return f"verifier: 'paris is the capital of germany' contradicts reference data", False
        if "everest" in text and "europe" in text:
            return f"verifier: 'mt everest is in europe' contradicts reference data", False
    if len([l for l in output.splitlines() if l.startswith("-")]) != 3:
        return "verifier: expected 3 bullet lines", False
    if any(len(l) > 60 for l in output.splitlines()):
        return "verifier: bullet exceeds 60 chars", False
    return "verifier: ok", True


def refine(topic: str, prev: str, critique: str, history: list[Attempt]) -> str:
    return generate(topic, history)


def run_loop(topic: str, use_critic: bool, max_iters: int = 4) -> list[Attempt]:
    history: list[Attempt] = []
    output = generate(topic, history)
    verify = verify_external if use_critic else (lambda o: feedback_self(o))
    for i in range(1, max_iters + 1):
        critique, ok = verify(output)
        history.append(Attempt(i, output, critique, ok))
        if ok:
            break
        output = refine(topic, output, critique, history)
    return history


def print_run(label: str, history: list[Attempt]) -> None:
    print(f"\n{label}")
    print("-" * 60)
    for a in history:
        tag = "OK " if a.verified else "..."
        print(f"  iter {a.iteration} {tag} critique: {a.critique}")
        for line in a.output.splitlines():
            print(f"    {line}")


def main() -> None:
    print("=" * 70)
    print("SELF-REFINE and CRITIC — Phase 14, Lesson 05")
    print("=" * 70)

    hist_self = run_loop("world facts", use_critic=False)
    print_run("Self-Refine (self-critique only)", hist_self)

    hist_critic = run_loop("world facts", use_critic=True)
    print_run("CRITIC (external verifier)", hist_critic)

    def summary(hist: list[Attempt]) -> str:
        return "passed" if hist and hist[-1].verified else "did not converge"

    print()
    print(f"Self-Refine ended: {summary(hist_self)}  after {len(hist_self)} iters")
    print(f"CRITIC    ended: {summary(hist_critic)}  after {len(hist_critic)} iters")
    print()
    print("Observation: CRITIC's verifier is grounded against reference data; a")
    print("self-critic can fail to flag its own confident-sounding hallucination.")


if __name__ == "__main__":
    main()