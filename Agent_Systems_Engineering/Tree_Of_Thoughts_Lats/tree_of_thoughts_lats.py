"""Toy Tree of Thoughts (BFS) and LATS (MCTS) on a stylized Game-of-24 search.

Mirrors the "Tree of Thoughts and LATS: Deliberate Search" lesson from
rohitg00/ai-engineering-from-scratch
(phases/14-agent-engineering/04-tree-of-thoughts-lats), whose starting point
is a critique of chain-of-thought: a single linear trajectory has no way to
back out of an early wrong step, so once the first "thought" is off track
the rest of the chain is too. Tree of Thoughts (Yao et al., NeurIPS 2023,
arXiv:2305.10601) reframes reasoning as search over a tree of intermediate
thoughts -- each node expands to several candidates, a value function scores
them, and BFS/DFS/beam search explores the highest-scoring branches. LATS
(Zhou et al., ICML 2024, arXiv:2310.04406) goes further and unifies ToT,
ReAct, and Reflexion under Monte Carlo Tree Search: a Policy proposes
actions (ReAct-style), a Value function scores partial trajectories
(ToT-style self-evaluation), and a Self-Reflector could write
natural-language diagnoses on failure (Reflexion-style -- not built here).
The papers report Game-of-24 accuracy going from 4% (GPT-4 CoT) to 74%
(ToT), and LATS hitting 92.7% pass@1 on HumanEval -- both without a single
gradient update.

Task, mirroring the paper's Game of 24 benchmark: given NUMBERS = [4, 6, 4,
1], combine all four with +, -, *, / (two at a time, three combine-steps
total) to reach TARGET = 24. Stdlib only, no LLM call anywhere -- `expand`
stands in for the Policy (every legal next thought, not just a handful
sampled), and `value` stands in for the Value function, scoring a partial
state by how close its remaining numbers are to TARGET.

Two search strategies run over the identical `expand`/`value` pair:

1. `tot_bfs` -- ToT's beam-flavored BFS: expand the whole frontier, score
               every child, keep only the top `max_expansions_per_level`,
               repeat for `max_depth` levels.
2. `mcts`    -- LATS' Select -> Expand -> Simulate -> Backpropagate loop,
               with `uct` balancing exploitation (`Node.q`) against
               exploration for the Select step.

Both are driven by the same `value` function, and that function has a real
blind spot worth knowing before reading the output: for a partial state
(more than one number left), it scores by whether ANY remaining number
already equals TARGET, not by whether the state is still solvable. A node
that happens to still be carrying a leftover exact-24 (e.g. from `6*4=24`,
with `4` and `1` left over) scores as well as a genuinely-finished node,
even though combining that leftover in can only move the total away from 24
unless the leftover is exactly the identity for its op (0 for +/-, 1 for
*/). Both algorithms fall for this in `main()`'s demo run: ToT's beam keeps
re-expanding "already has a 24 sitting in the state" children instead of the
multi-step branch that actually reaches 24 (`6+1=7, 7*4=28, 28-4=24` is
reachable -- an unpruned search finds it -- but it never survives the
top-8 cut), and LATS reports a best leaf sitting at `(24, 4)` with
`value == 0.000`, which reads as "solved" but is a dead end: the puzzle
isn't over (two numbers still need combining), and no op on `24` and `4`
returns to 24. See the README for the traced-through walkthrough of exactly
where each search gets misled.

A second, narrower limitation: `expand` only ever computes `a - b` and
`a / b` for `a >= b` (state is kept sorted descending, so `i < j` in
`itertools.combinations` always means `state[i] >= state[j]`), never the
reverse order. Some valid Game-of-24 solutions need a negative intermediate
(e.g. `4 - (4 * (1 - 6))` needs `1 - 6`) and are structurally unreachable
here -- though for this specific NUMBERS, order-preserving solutions still
exist and are what the unpruned search above finds.

Run directly (`python tree_of_thoughts_lats.py`) to reproduce the demo.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field


NUMBERS = [4, 6, 4, 1]
TARGET = 24
OPS = ["+", "-", "*", "/"]


# --- 1. Search-tree node & arithmetic step ---------------------------------------------------
@dataclass
class Node:
    """One state in the search tree: a partial or final arrangement of numbers, plus MCTS bookkeeping.

    Attributes:
        state: Remaining numbers, sorted descending. Length 4 at the root,
            shrinking by one every combine-step; length 1 at a terminal
            node.
        trace: The `"a op b=result"` strings recording how this state was
            reached from the root, in order.
        visits: How many times `mcts`'s backprop has touched this node.
            Unused by `tot_bfs`, which never reads or writes it.
        value_sum: Running total of rewards backpropagated through this
            node; `q` is this divided by `visits`.
        children: Populated by `expand` once a node has been expanded;
            empty means "not yet expanded" as much as "no legal moves
            left" -- both `select`/`mcts`'s while-loop and `_all_leaves`
            treat an empty list as a leaf.
    """

    state: tuple[float, ...]
    trace: list[str]
    visits: int = 0
    value_sum: float = 0.0
    children: list["Node"] = field(default_factory=list)

    @property
    def q(self) -> float:
        """Mean backpropagated reward, `value_sum / visits`, or `0.0` before the first visit."""
        return self.value_sum / self.visits if self.visits else 0.0


def evaluate(a: float, op: str, b: float) -> float | None:
    """Apply `op` to `a, b`, or return `None` if the result isn't a valid number.

    Only `/` can fail (division by zero); `+`, `-`, `*` always succeed.
    `expand` drops any child whose `evaluate` call returns `None`, so a
    division by zero simply prunes that one candidate rather than raising.
    """
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b != 0 else None
    return None


# --- 2. Policy: thought generation ------------------------------------------------------------
def expand(node: Node) -> list[Node]:
    """Generate every legal one-step child of `node`: the ToT/LATS Policy step.

    Picks every unordered pair of positions in `node.state` (there are no
    legal moves once fewer than two numbers remain, hence the early
    return), applies all four ops to each pair, and replaces the pair with
    the result -- re-sorted descending, so every later `expand` call keeps
    seeing `state[i] >= state[j]` for `i < j`. This is a real restriction,
    not just a convention: `evaluate(a, op, b)` is only ever called with
    `a >= b`, so `a - b` and `a / b` are the only orders ever tried --
    `b - a` and `b / a` never are (see the module docstring).

    A duplicate input value (`NUMBERS` has two `4`s) also means two
    distinct index-pairs can produce numerically identical children --
    e.g. combining `6` with "the first 4" versus "the second 4" both yield
    a `6*4=24` node with the same resulting `state`. `expand` does not
    deduplicate these; both survive as separate `Node` objects and can
    each occupy a slot in `tot_bfs`'s beam.

    Returns:
        One child `Node` per (position pair, op) combination that produced
        a real number, each with `visits=0` and no children of its own
        yet. Empty once `len(node.state) < 2`.
    """
    children: list[Node] = []
    state = node.state
    if len(state) < 2:
        return children
    for i, j in itertools.combinations(range(len(state)), 2):
        for op in OPS:
            a, b = state[i], state[j]
            v = evaluate(a, op, b)
            if v is None:
                continue
            remaining = [s for k, s in enumerate(state) if k not in (i, j)]
            new_state = tuple(sorted(remaining + [v], reverse=True))
            step = f"{a}{op}{b}={v}"
            children.append(Node(state=new_state, trace=node.trace + [step]))
    return children


# --- 3. Value function (self-evaluation stand-in) ---------------------------------------------
def value(node: Node) -> float:
    """Score `node`: the stand-in for the ToT/LATS Value function's self-evaluation.

    Terminal (`len(state) == 1`): `1.0` if the single remaining number is
    within `1e-6` of TARGET, else the negative distance scaled down
    (`-abs(result - TARGET) / 100.0`) so terminal misses never accidentally
    outrank a non-terminal node.

    Non-terminal: `-best_distance / 100.0`, where `best_distance` is the
    smallest distance from TARGET across ALL remaining numbers -- not
    whether the state is still solvable. This is the function's real blind
    spot: a state that still happens to contain an exact TARGET among its
    leftovers (e.g. `(24, 4, 1)`, from an early `6*4=24`) scores `0.0`,
    tied with a node that is one identity-op away from actually finishing,
    even though the leftover numbers can only be combined into that 24 via
    a non-identity op that moves it away from TARGET. Both `tot_bfs` and
    `mcts` get misled by exactly this in `main()`'s demo run (see the
    module docstring and the README).
    """
    if len(node.state) == 1:
        result = node.state[0]
        return 1.0 if abs(result - TARGET) < 1e-6 else -abs(result - TARGET) / 100.0
    best_distance = min(abs(v - TARGET) for v in node.state)
    return -best_distance / 100.0


# --- 4. Tree of Thoughts: beam-flavored BFS ----------------------------------------------------
def tot_bfs(root: Node, max_expansions_per_level: int = 8,
            max_depth: int = 3) -> tuple[Node | None, int]:
    """ToT's beam-flavored BFS: expand the frontier, score every child, keep the top `max_expansions_per_level`, repeat.

    Returns the first child scored above `0.99` the moment it's generated
    -- before that level's pruning happens, so an exact match right at the
    edge of the frontier is never at risk of being cut. If no child ever
    crosses that bar within `max_depth` levels, returns the single
    best-scoring node left in the final frontier by direct comparison,
    ignoring how it got there.

    Args:
        root: The starting node (root of the search).
        max_expansions_per_level: Beam width -- how many children survive
            each level's prune. `value`'s blind spot (see module
            docstring) means this beam can fill up with "already looks
            solved" false positives that crowd out the real solution
            branch; this is exactly what happens in the demo (see
            README).
        max_depth: Number of expand-and-prune rounds to run. `NUMBERS` has
            4 entries, so a fully-reduced terminal node is reachable in
            exactly 3 combine-steps; `max_depth=3` is the minimum that can
            reach one.

    Returns:
        `(best_node, expansions)`: the best node found (`None` only if
        `root` already has fewer than two numbers, so the loop's frontier
        is instantly empty) and the total number of children generated
        across all levels, irrespective of how many survived pruning.
    """
    frontier = [root]
    expansions = 0
    for _ in range(max_depth):
        scored: list[tuple[float, Node]] = []
        for node in frontier:
            for child in expand(node):
                expansions += 1
                scored.append((value(child), child))
                if value(child) > 0.99:
                    return child, expansions
        scored.sort(key=lambda p: p[0], reverse=True)
        frontier = [n for _, n in scored[:max_expansions_per_level]]
    best = max(frontier, key=value) if frontier else None
    return best, expansions


# --- 5. LATS: Monte Carlo Tree Search -----------------------------------------------------------
def uct(parent: Node, child: Node, c: float = 1.4) -> float:
    """Upper Confidence bound for Trees: `child`'s exploitation/exploration score for `select`'s argmax.

    Returns `float("inf")` for an unvisited child so `select`/`mcts` always
    try every child at least once before trusting `q`. Otherwise
    `child.q + c * sqrt(log(parent.visits) / child.visits)`: the first
    term exploits the child's mean backpropagated reward, the second
    explores in proportion to how rarely it's been visited relative to its
    parent; `c` trades one off against the other (a higher `c` favors
    under-visited children more).
    """
    if child.visits == 0:
        return float("inf")
    return child.q + c * math.sqrt(math.log(parent.visits) / child.visits)


def select(node: Node) -> Node:
    """Descend from `node` to a leaf by always following the highest-`uct` child: MCTS's Select phase in isolation.

    Not called anywhere in this module -- `mcts` needs the full
    root-to-leaf path for `backprop`, so it inlines this exact same
    while-loop itself rather than calling this function, which only
    returns the leaf and discards the path it walked to get there. Kept as
    the single-purpose reference implementation of "Select" on its own,
    decoupled from path bookkeeping; not exercised by `main()`'s demo.
    """
    while node.children:
        node = max(node.children, key=lambda ch: uct(node, ch))
    return node


def simulate(node: Node, depth: int, rng: random.Random) -> float:
    """Random rollout from `node`, `depth` steps deep (or until no legal moves remain), then score the endpoint.

    `depth` is chosen by the caller as `3 - len(node.trace)` -- exactly how
    many combine-steps remain before `NUMBERS`' four numbers are reduced to
    one -- so a rollout that doesn't run out of `expand` options first
    lands on a genuine terminal state. Each step picks uniformly among
    `expand`'s candidates via `rng.choice`, so two calls sharing an `rng`
    can diverge from here on.

    Returns:
        `value` of wherever the rollout stopped (terminal, or non-terminal
        if it ran out of steps or `expand` returned nothing early).
    """
    current = node
    for _ in range(depth):
        options = expand(current)
        if not options:
            break
        current = rng.choice(options)
    return value(current)


def backprop(path: list[Node], reward: float) -> None:
    """Add `reward` to every node in `path`'s `value_sum` and increment each one's `visits` by one: MCTS's Backpropagate phase."""
    for n in path:
        n.visits += 1
        n.value_sum += reward


def mcts(root: Node, iterations: int, rng: random.Random) -> tuple[Node, int]:
    """LATS' Select -> Expand -> Simulate -> Backpropagate loop, run for `iterations` rounds.

    Each iteration: descend from `root` via `uct` to a leaf (inlining
    `select`'s logic so the full path is available for `backprop`); if
    that leaf has already been visited once before (`visits > 0`) and
    still has more than one number left, expand it
    (`leaf.children = expand(leaf)`) and step into its FIRST generated
    child specifically -- not a random or best-scoring one, just
    `children[0]` -- so only that one child's branch gets a
    simulate/backprop this iteration, while its siblings start out as
    plain `visits=0` leaves for a later iteration's `uct` (which scores an
    unvisited child as `inf`) to pick up. Then roll out from wherever the
    descent landed via `simulate`, and backpropagate the resulting reward
    along the whole path.

    Args:
        root: Root node. `main()` pre-expands it once before calling
            `mcts` (`root.children = expand(root)`) so the very first
            iteration's descent has somewhere to go.
        iterations: Number of Select/Expand/Simulate/Backpropagate rounds.
        rng: Source of randomness for `simulate`'s rollouts; a fixed seed
            makes an entire `mcts` call reproducible.

    Returns:
        `(best_leaf, expansions)`: `best_leaf` is chosen from every
        current leaf in the tree (`_all_leaves`) by `value` alone -- the
        node's own symbolic score, not its accumulated `q` or `visits` --
        so a leaf that was only ever touched once by a lucky rollout can
        outrank a heavily-visited one; this is why the demo's reported
        LATS answer can look "solved" (see module docstring) without
        actually being finished. `expansions` counts children created by
        every `expand(leaf)` call made during the loop (not counting the
        one `main()` does before calling `mcts`).
    """
    expansions = 0
    for _ in range(iterations):
        path = [root]
        cur = root
        while cur.children:
            cur = max(cur.children, key=lambda ch: uct(cur, ch))
            path.append(cur)
        if cur.visits > 0 and len(cur.state) > 1:
            cur.children = expand(cur)
            expansions += len(cur.children)
            if cur.children:
                cur = cur.children[0]
                path.append(cur)
        reward = simulate(cur, depth=max(0, 3 - len(cur.trace)), rng=rng)
        backprop(path, reward)
    best_leaf = max(_all_leaves(root), key=value, default=root)
    return best_leaf, expansions


def _all_leaves(node: Node) -> list[Node]:
    """Collect every node in the subtree rooted at `node` with no children (never expanded), depth-first."""
    if not node.children:
        return [node]
    out: list[Node] = []
    for ch in node.children:
        out.extend(_all_leaves(ch))
    return out


# --- 6. Reporting & demo ------------------------------------------------------------------------
def main() -> None:
    """Run ToT BFS and LATS MCTS on the same Game-of-24 instance and print both traces plus the paper's headline numbers.

    With `NUMBERS = [4, 6, 4, 1]`, `TARGET = 24`, and a fixed
    `random.Random(7)` seed for LATS, this reproduces deterministically:
    ToT lands on `(27,)` (trace `['6*4=24', '4-1=3', '24+3=27']`,
    `value=-0.030`), and LATS lands on `(24, 4)` (trace
    `['6*4=24', '24*1=24']`, `value=0.000`) -- neither actually reaches the
    exact-24 solution that exists in the reachable search space (e.g.
    `6+1=7, 7*4=28, 28-4=24`), for the reasons documented on `value` and in
    the module docstring.
    """
    print("=" * 70)
    print("TREE OF THOUGHTS + LATS — Phase 14, Lesson 04")
    print("=" * 70)
    print(f"numbers: {NUMBERS}  target: {TARGET}")

    root_tot = Node(state=tuple(sorted(NUMBERS, reverse=True)), trace=[])
    best_tot, n_tot = tot_bfs(root_tot)
    print("\nToT BFS")
    print("-" * 60)
    if best_tot is not None:
        print(f"  best trace: {best_tot.trace}")
        print(f"  final state: {best_tot.state}  value: {value(best_tot):.3f}")
    print(f"  expansions: {n_tot}")

    rng = random.Random(7)
    root_lats = Node(state=tuple(sorted(NUMBERS, reverse=True)), trace=[])
    root_lats.children = expand(root_lats)
    for ch in root_lats.children:
        ch.visits = 0
    best_lats, n_lats = mcts(root_lats, iterations=80, rng=rng)
    print("\nLATS MCTS")
    print("-" * 60)
    print(f"  best trace: {best_lats.trace}")
    print(f"  final state: {best_lats.state}  value: {value(best_lats):.3f}")
    print(f"  node expansions: {n_lats}")

    print()
    print("Paper headlines (for reference):")
    print("  ToT Game-of-24:  GPT-4 CoT 4%  -> ToT 74%")
    print("  LATS HumanEval:  pass@1 92.7% with GPT-4 (SOTA at paper time)")
    print("  Cost: ToT uses 100-1000x the tokens of CoT. Use with intent.")


if __name__ == "__main__":
    main()
