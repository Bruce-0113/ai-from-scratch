"""A stdlib actor runtime modeled on AutoGen v0.4's Core layer: agents as
actors with private state and an inbox, exchanging typed messages as the
only form of interaction, with the runtime -- not the actors themselves --
deciding what happens when a handler raises.

Mirrors the "The Actor Model for Agents -- Async Messages and Typed
Runtimes" lesson from rohitg00/ai-engineering-from-scratch
(phases/14-agent-engineering/14-autogen-actor-model), whose central claim
is that most agent frameworks are synchronous: one agent produces, one
consumes, in a single call stack, so a failure crashes the stack and
concurrency has to be bolted on afterward. AutoGen v0.4's redesign
(Microsoft Research, Jan 2025) answers this with the actor model:

- An actor has private state (never touched directly from outside), an
  inbox (message queue), and a handler `receive(message, runtime)` whose
  effects can be "reply," "send to another actor," "spawn," "update
  state," or "stop self." Two actors interact only by exchanging
  messages, never by sharing memory.
- Decoupling *delivery* from *handling* is what buys three properties:
  fault isolation (one actor's crash doesn't crash another -- the runtime
  catches it and decides whether to log, retry, or dead-letter it),
  natural concurrency (many messages can be in flight at once), and
  distribution-readiness (inbox + transport is the same abstraction
  whether the actor is in-process or on another host).
- AutoGen v0.4 splits its surface into three API layers: Core (the
  low-level actor framework -- `AgentRuntime`, `Agent`, `Message`,
  `Topic`), AgentChat (task-driven, e.g. `RoundRobinGroupChat` /
  `SelectorGroupChat`), and Extensions (integrations). This module only
  implements Core's shape -- a single-process runtime, not a real
  scheduler or transport.

Two parts: a small actor runtime (`Message`, `Actor`, `Runtime`) and a
two-actor demo (`ReviewerAgent`, `ChecklistAgent`) that exchange messages
until they reach a consensus verdict, with one message deliberately
crashing its handler to show that the other actor keeps running.

1. Message                          -- typed envelope: sender, recipient,
                                        topic, body, id
2. Actor                            -- abstract base; subclasses implement
                                        `receive`
3. Runtime                          -- event loop: queue, delivery, fault
                                        isolation, trace
4. ReviewerAgent / ChecklistAgent   -- demo actors exchanging review
                                        messages to consensus
5. main                             -- wires up the runtime, sends the
                                        opening messages, prints the trace

Run directly (`python actor_model_for_agent.py`) to reproduce the demo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


# --- 1. Message envelope ----------------------------------------------------------------
@dataclass
class Message:
    """A typed envelope -- the only way two actors interact.

    Attributes:
        sender: Name of the actor (or `"__user__"`) that sent this message.
        recipient: Name of the actor this message is addressed to.
        topic: Label the receiving actor's `receive` dispatches on.
        body: Arbitrary payload; shape is a convention between sender and
            recipient, not enforced by the runtime.
        mid: Monotonically increasing message id, assigned by
            `Runtime.send`; used only for tracing.
    """
    sender: str
    recipient: str
    topic: str
    body: Any
    mid: int = 0


# --- 2. Actor base class -----------------------------------------------------------------
class Actor:
    """Abstract actor: a name plus a message handler.

    An actor's state lives on `self` and is never touched from outside --
    the only way another actor affects it is by sending a message that
    `receive` chooses to act on.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def receive(self, message: Message, runtime: "Runtime") -> None:
        """Handle one delivered message.

        Args:
            message: The message being delivered to this actor.
            runtime: The runtime delivering it, so a handler can `send`
                further messages (e.g. a reply) as a side effect.

        Raises:
            NotImplementedError: Subclasses must override this.
        """
        raise NotImplementedError


# --- 3. Runtime: queue, delivery, fault isolation, trace ---------------------------------
@dataclass
class Runtime:
    """Single-process event loop that owns delivery, not handling.

    `send` only enqueues a message and returns immediately -- it never
    calls a handler directly. `run_until_idle` is what actually delivers
    messages, one at a time, catching any exception a handler raises so
    that one actor's failure never stops another actor's messages from
    being processed.

    Attributes:
        actors: Registered actors, keyed by name.
        queue: Pending messages, delivered in FIFO order.
        dead_letters: `(message, reason)` pairs for messages that could not
            be delivered (unknown recipient) or whose handler raised.
        counter: Source of monotonically increasing `Message.mid` values.
        trace: Human-readable log of every send, receive, and failure, in
            the order they happened.
        max_messages: Safety cap on how many messages `run_until_idle`
            will process, in case actors keep sending to each other.
    """
    actors: dict[str, Actor] = field(default_factory=dict)
    queue: deque[Message] = field(default_factory=deque)
    dead_letters: list[tuple[Message, str]] = field(default_factory=list)
    counter: int = 0
    trace: list[str] = field(default_factory=list)
    max_messages: int = 100

    def register(self, actor: Actor) -> None:
        """Add `actor` to the runtime, addressable by `actor.name`."""
        self.actors[actor.name] = actor

    def send(self, sender: str, recipient: str, topic: str, body: Any) -> None:
        """Enqueue a message for later delivery.

        This is the "decouple delivery from handling" step: `send` never
        invokes `recipient`'s handler itself, it just appends to the queue
        and records the send in `trace`. Delivery happens later, inside
        `run_until_idle`.

        Args:
            sender: Name of the sending actor (or `"__user__"`).
            recipient: Name of the actor the message is addressed to.
            topic: Label the recipient's `receive` will dispatch on.
            body: Arbitrary payload for the recipient to interpret.
        """
        self.counter += 1
        msg = Message(sender=sender, recipient=recipient,
                      topic=topic, body=body, mid=self.counter)
        self.queue.append(msg)
        self.trace.append(
            f"[send m{msg.mid:03d}] {sender} -> {recipient} topic={topic} body={body}"
        )

    def run_until_idle(self) -> None:
        """Deliver queued messages one at a time until the queue is empty.

        Every delivery is wrapped in a `try/except`: if a handler raises,
        the exception is recorded in `dead_letters` and `trace` instead of
        propagating, so the loop moves on to the next message -- this is
        the fault-isolation property the actor model buys. A message
        addressed to an unregistered actor is dead-lettered the same way,
        without ever calling a handler. `max_messages` bounds the loop in
        case handlers keep sending new messages to each other.
        """
        processed = 0
        while self.queue and processed < self.max_messages:
            msg = self.queue.popleft()
            actor = self.actors.get(msg.recipient)
            if actor is None:
                self.dead_letters.append((msg, f"no actor {msg.recipient!r}"))
                self.trace.append(f"[DLQ m{msg.mid:03d}] no actor {msg.recipient!r}")
                continue
            try:
                actor.receive(msg, self)
                self.trace.append(
                    f"[recv m{msg.mid:03d}] {actor.name} handled topic={msg.topic}"
                )
            except Exception as e:
                self.dead_letters.append((msg, f"{type(e).__name__}: {e}"))
                self.trace.append(
                    f"[FAIL m{msg.mid:03d}] {actor.name} raised "
                    f"{type(e).__name__}: {e}  (others keep running)"
                )
            processed += 1


# --- 4. Demo actors: a reviewer and a checklist that drives it ---------------------------
class ReviewerAgent(Actor):
    """Reviews code snippets for two hard-coded hazards, and can be told to crash.

    Attributes:
        verdicts: `(code, ok)` pairs recorded for every `"review"` message
            handled, in order.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.verdicts: list[tuple[str, bool]] = []

    def receive(self, message: Message, runtime: Runtime) -> None:
        """Dispatch on `message.topic`: review a snippet, or simulate a crash.

        On `"review"`, checks `message.body` for `eval(` and a bare
        `except:`, records the verdict, and replies to `message.sender`
        with a `"review_result"` message. On `"crash_me"`, raises
        unconditionally -- `Runtime.run_until_idle` catches this and
        dead-letters it without affecting any other actor.
        """
        if message.topic == "review":
            code = str(message.body)
            issues = []
            if "eval(" in code:
                issues.append("uses eval")
            if "except:" in code:
                issues.append("bare except")
            ok = len(issues) == 0
            self.verdicts.append((code, ok))
            runtime.send(
                sender=self.name,
                recipient=message.sender,
                topic="review_result",
                body={"ok": ok, "issues": issues},
            )
        elif message.topic == "crash_me":
            raise RuntimeError("simulated handler failure")


class ChecklistAgent(Actor):
    """Fans a batch of snippets out to a reviewer, then waits for consensus.

    Attributes:
        partner: Name of the `ReviewerAgent` this checklist sends
            `"review"` requests to.
        results: One review-result dict per reply received so far.
        consensus: `None` until all expected replies are in; then `True`
            only if every result was `ok`.
    """

    def __init__(self, name: str, partner: str) -> None:
        super().__init__(name)
        self.partner = partner
        self.results: list[dict[str, Any]] = []
        self.consensus: bool | None = None

    def receive(self, message: Message, runtime: Runtime) -> None:
        """Dispatch on `message.topic`: kick off reviews, or tally a result.

        On `"start"`, sends one `"review"` message to `self.partner` per
        snippet in `message.body`. On `"review_result"`, appends it to
        `self.results` and updates `self.consensus`; the final value is
        only meaningful once every reply for the batch has arrived (this
        demo hard-codes that count to 3, matching the `"start"` batch size
        sent in `main`).
        """
        if message.topic == "start":
            for snippet in message.body:
                runtime.send(
                    sender=self.name, recipient=self.partner,
                    topic="review", body=snippet,
                )
        elif message.topic == "review_result":
            self.results.append(dict(message.body))
            if all(r["ok"] for r in self.results):
                self.consensus = True
            if len(self.results) == 3:
                self.consensus = all(r["ok"] for r in self.results)


# --- 5. Demo -------------------------------------------------------------------------------
def main() -> None:
    """Wire up one reviewer and one checklist actor, run to idle, print the trace.

    Sends a batch of three snippets to review (one clean, one using
    `eval`, one with a bare `except`) plus a `"crash_me"` message straight
    to the reviewer, then drains the queue and prints the message trace,
    the checklist's consensus verdict, and the dead-letter queue -- so the
    fault-isolation property (the crash doesn't block the reviews) is
    visible directly in the output.
    """
    print("=" * 70)
    print("AUTOGEN V0.4 ACTOR RUNTIME (STDLIB) — Phase 14, Lesson 14")
    print("=" * 70)

    runtime = Runtime()
    reviewer = ReviewerAgent("reviewer")
    checklist = ChecklistAgent("checklist", partner="reviewer")
    runtime.register(reviewer)
    runtime.register(checklist)

    runtime.send(
        sender="__user__",
        recipient="checklist",
        topic="start",
        body=[
            "def add(a, b): return a + b",
            "def hazard(): eval('1+1')",
            "def silent(): \n    try:\n        f()\n    except:\n        pass",
        ],
    )

    runtime.send(
        sender="__user__",
        recipient="reviewer",
        topic="crash_me",
        body={},
    )

    runtime.run_until_idle()

    print("\nmessage trace")
    for line in runtime.trace:
        print(f"  {line}")

    print(f"\nchecklist consensus: {checklist.consensus}")
    print(f"dead-letter queue:   {len(runtime.dead_letters)} message(s)")
    for msg, reason in runtime.dead_letters:
        print(f"  DLQ m{msg.mid:03d} ({reason}) "
              f"{msg.sender} -> {msg.recipient} topic={msg.topic}")

    print()
    print("property: reviewer's crash on 'crash_me' did not stop")
    print("the 'review' messages from being processed. fault isolation.")


if __name__ == "__main__":
    main()
