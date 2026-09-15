"""Toy Reflexion loop -- Actor, Evaluator, Self-Reflector, Episodic memory. Stdlib only.

Mirrors the "Reflexion: Verbal Reinforcement Learning" lesson from
rohitg00/ai-engineering-from-scratch (phases/14-agent-engineering/03-
reflexion-verbal-rl), whose starting point is a critique of gradient-based
RL: fixing a failure mode by updating weights needs thousands of trials and
a GPU cluster, and most production agents don't have a training budget for
every failure. Reflexion (Shinn et al., arXiv:2303.11366) bets that an agent
can improve between trials with no weight update at all -- just write a
natural-language reflection on why the last trial failed, store it, and
condition the next trial on it. The paper reports beating ReAct on ALFWorld
and HotpotQA, and setting state of the art (at the time) on HumanEval/MBPP
code generation, all without a single gradient step.

This module builds the four pieces directly, on a toy task: pick three
integers from 1..9 that sum to TARGET.

1. Reflection / EpisodicMemory -- one reflection is a (trial number, text)
                                   pair; EpisodicMemory is a bounded FIFO
                                   buffer of them, rendered to a prompt
                                   string by `as_prompt`
2. Actor                        -- stands in for the real Actor LLM call;
                                   scripted here to start with a bad guess
                                   and move toward the target once it can
                                   see reflections in memory
3. binary_evaluator              -- the lesson's "scalar" evaluator type:
                                   ground-truth pass/fail plus a signed
                                   distance from the target
4. SelfReflector                 -- stands in for the real Self-Reflector
                                   LLM call; scripted templates instead of
                                   free text, keyed off the evaluator's
                                   signed distance
5. TrialResult / run_reflexion   -- one trial's full record, and the loop
                                   that runs Actor -> Evaluator -> Self-
                                   Reflector, storing a reflection and
                                   retrying until success or max_trials
6. summarize / main              -- prints a `use_memory=False` baseline
                                   next to a `use_memory=True` run on the
                                   same task, so the effect of conditioning
                                   on reflections is visible trial by trial

Run directly (`python reflexion_verbal_rl.py`) to reproduce the demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TARGET = 20


# --- 1. Episodic memory: bounded buffer of reflections ------------------------------------
@dataclass
class Reflection:
    """One entry in episodic memory: the self-reflection written after a failed trial.

    Attributes:
        trial: The trial number this reflection was written after (1-based).
        text: The natural-language diagnosis from `SelfReflector.reflect`.
    """

    trial: int
    text: str


@dataclass
class EpisodicMemory:
    """Bounded FIFO buffer of `Reflection`s for one task, rendered to a prompt string.

    Attributes:
        items: Reflections in the order they were added.
        max_len: Oldest reflection is dropped once `items` would exceed this
            length -- a crude stand-in for the lesson's memory-rot
            mitigations (TTL / compaction), never actually triggered by this
            demo's short runs (at most two reflections are ever stored).
    """

    items: list[Reflection] = field(default_factory=list)
    max_len: int = 6

    def add(self, r: Reflection) -> None:
        """Append `r`, then drop the oldest entry if `items` is over `max_len`."""
        self.items.append(r)
        if len(self.items) > self.max_len:
            self.items.pop(0)

    def as_prompt(self) -> str:
        """Render `items` as a bullet list, oldest first, for prepending to a prompt.

        Returns "(no prior reflections)" when empty. This is what a real,
        LLM-driven Actor would prepend to its next-trial prompt; the
        scripted `Actor` below never calls it -- it only checks
        `len(memory.items)`, not what any reflection actually says.
        """
        if not self.items:
            return "(no prior reflections)"
        lines = [f"- trial {r.trial}: {r.text}" for r in self.items]
        return "\n".join(lines)


# --- 2. Actor (stand-in for the real Actor LLM call) ---------------------------------------
class Actor:
    """Scripted stand-in for the real Actor LLM call.

    Chooses its attempt purely from how many reflections are in memory --
    `len(memory.items)` -- never from what any reflection actually says.
    Reflection *count* proxies for "the actor has now seen this much
    feedback," so the demo can show memory changing behavior without a
    real model reading any text.
    """

    def act(self, memory: EpisodicMemory) -> list[int]:
        """Return a fixed attempt keyed off `len(memory.items)`.

        No reflections seen (`n == 0`): guesses low, `[1, 2, 3]`. One
        reflection seen: corrects toward the target, `[5, 6, 7]`. Two or
        more: converges on the target-summing `[6, 7, 7]`. Given
        `TARGET == 20` and `run_reflexion`'s stop-on-first-success loop,
        the `n == 2` branch always succeeds before `n` can reach 3, so the
        final `return` is unreachable from any call in this module's
        `main` -- it only guarantees the function has a defined result if
        `run_reflexion` were ever driven past three failed trials with
        memory on.
        """
        n = len(memory.items)
        if n == 0:
            return [1, 2, 3]
        if n == 1:
            return [5, 6, 7]
        if n == 2:
            return [6, 7, 7]
        return [6, 7, 7]


# --- 3. Evaluator (scalar / binary) ---------------------------------------------------------
def binary_evaluator(attempt: list[int], target: int) -> tuple[bool, int]:
    """Score `attempt` against `target`: the lesson's "scalar" evaluator type.

    Returns:
        A `(success, delta)` pair: `success` is exact-match pass/fail,
        `delta` is `sum(attempt) - target` (negative when short, positive
        when over, zero on success). Only `success` drives the loop's stop
        condition; `delta`'s sign is what `SelfReflector.reflect` reads to
        pick which templated reflection to write.
    """
    total = sum(attempt)
    return total == target, total - target


# --- 4. Self-Reflector (stand-in for the real Self-Reflector LLM call) ---------------------
class SelfReflector:
    """Scripted stand-in for the real Self-Reflector LLM call."""

    def reflect(self, attempt: list[int], delta: int) -> str:
        """Return a one-line templated diagnosis, chosen by the sign of `delta`.

        A real Self-Reflector would read the trajectory and write free-form
        prose ("I picked the wrong tool because..."); this substitutes
        three fixed templates keyed off whether `attempt` undershot,
        overshot, or hit the target exactly, so the demo stays
        deterministic and model-free.
        """
        if delta < 0:
            return f"sum {sum(attempt)} is {-delta} short; pick larger values"
        if delta > 0:
            return f"sum {sum(attempt)} overshoots by {delta}; pick smaller values"
        return "succeeded"


# --- 5. One trial's record, and the Actor -> Evaluator -> Self-Reflector loop ---------------
@dataclass
class TrialResult:
    """Everything one trial produced: the guess, the verdict, and the reflection.

    Attributes:
        trial: 1-based trial number.
        attempt: The three integers the Actor guessed.
        success: Whether `attempt` summed to `TARGET`.
        delta: Signed distance from `TARGET` (see `binary_evaluator`).
        reflection: The text `SelfReflector.reflect` wrote for this trial.
    """

    trial: int
    attempt: list[int]
    success: bool
    delta: int
    reflection: str


def run_reflexion(max_trials: int, use_memory: bool) -> list[TrialResult]:
    """Run the Actor -> Evaluator -> Self-Reflector loop for up to `max_trials` trials.

    Every trial is scored and reflected on regardless of `use_memory` -- the
    same `memory` object accumulates a `Reflection` after every failed
    trial. What `use_memory` actually controls is what the *Actor* is
    allowed to see: when `True`, the Actor gets the real, accumulating
    `memory`; when `False`, it gets a brand-new, always-empty
    `EpisodicMemory()` instance on every single trial, so it can never
    observe that any reflection was ever written -- even though one was.
    This is what makes `use_memory=False` a genuine "no conditioning on
    reflections" baseline, rather than "no reflections are generated at
    all."

    Stops early the first time a trial succeeds; otherwise runs the full
    `max_trials` trials and returns however many were attempted.

    Args:
        max_trials: Upper bound on trials to run.
        use_memory: Whether the Actor sees the accumulating episodic
            memory (`True`) or a fresh, empty one every trial (`False`).

    Returns:
        One `TrialResult` per trial actually run, in order.
    """
    actor = Actor()
    reflector = SelfReflector()
    memory = EpisodicMemory()
    trials: list[TrialResult] = []
    for t in range(1, max_trials + 1):
        attempt = actor.act(memory if use_memory else EpisodicMemory())
        success, delta = binary_evaluator(attempt, TARGET)
        text = reflector.reflect(attempt, delta)
        trials.append(TrialResult(t, attempt, success, delta, text))
        if success:
            break
        memory.add(Reflection(trial=t, text=text))
    return trials


# --- 6. Reporting & demo ---------------------------------------------------------------------
def summarize(trials: list[TrialResult], name: str) -> None:
    """Print `trials` under a `name` header, one line per trial plus a final verdict."""
    print(f"\n{name}")
    print("-" * 60)
    for r in trials:
        mark = "OK " if r.success else "..."
        print(f"  trial {r.trial}: {r.attempt} sum={sum(r.attempt)} "
              f"delta={r.delta:+d} {mark} -> {r.reflection}")
    last = trials[-1]
    print(f"  final: {'success' if last.success else 'failed'} "
          f"at trial {last.trial}")


def main() -> None:
    """Run the target-sum demo with and without episodic memory, and compare trial counts.

    Runs `run_reflexion` twice on the same task (`max_trials=4`): once with
    `use_memory=False` (baseline -- the Actor never sees a reflection, so
    it repeats its first guess every trial and never succeeds) and once
    with `use_memory=True` (the Actor sees one more reflection each trial
    and converges by trial 3). Prints both trial-by-trial traces via
    `summarize`, then the trial counts each run used.
    """
    print("=" * 70)
    print(f"REFLEXION — pick three ints in [1..9] summing to {TARGET}")
    print("Phase 14, Lesson 03")
    print("=" * 70)

    trials_no_mem = run_reflexion(max_trials=4, use_memory=False)
    summarize(trials_no_mem, "BASELINE (no episodic memory)")

    trials_mem = run_reflexion(max_trials=4, use_memory=True)
    summarize(trials_mem, "REFLEXION (episodic memory on)")

    baseline_steps = len(trials_no_mem)
    reflex_steps = len(trials_mem)
    print()
    print(f"baseline used {baseline_steps} trials; reflexion used {reflex_steps}.")
    print("Without a reflection in the prompt, the scripted actor never adapts.")
    print("With one reflection, the actor corrects; with two, it converges.")


if __name__ == "__main__":
    main()
