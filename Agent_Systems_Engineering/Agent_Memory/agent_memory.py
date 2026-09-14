"""MemGPT-shaped two-tier memory in stdlib -- no LLM call, a scripted agent
drives the tool calls so the paging control flow is testable offline.

Mirrors the "Agent Memory -- Virtual Context and Memory Paging" lesson from
rohitg00/ai-engineering-from-scratch
(phases/14-agent-engineering/07-memory-virtual-context-memgpt), which maps
MemGPT's design onto OS virtual memory: main context is RAM (always visible,
fixed size), archival memory is disk (unbounded, paged in on demand), and a
memory tool call is a page fault -- the agent "blocks", the runtime resolves
the read/write, and the result splices back into the next turn as a new
observation, the same shape as a Unix `read()` syscall.

Context windows look like they should solve memory; they don't -- long-
horizon agents still need facts that fall outside even a 128k window once
turns pile up (dilution) or a new session starts from empty (no
persistence). MemGPT's answer is two tiers plus five tools that move facts
between them:

1. Message / MainContext          -- RAM: a fixed-size prompt buffer (core
                                      dict of persistent sections + a
                                      rolling message list); oldest messages
                                      are auto-evicted once the list
                                      overflows, not dropped
2. ArchivalRecord / ArchivalStore  -- disk: an unbounded external store,
                                      searchable by token overlap (a toy
                                      stand-in for BM25/embedding retrieval)
3. MemoryTools                     -- the five canonical memory tools:
                                      append/replace core memory, insert/
                                      search archival memory, search past
                                      conversation turns
4. ToolCall / run_scripted_agent   -- a scripted stand-in for the agent's
                                      control loop; no LLM call, so the
                                      paging behavior is deterministic and
                                      testable
5. main                            -- fills main context until eviction
                                      kicks in, then pages the evicted fact
                                      back in via archival search and
                                      conversation search

Run directly (`python agent_memory.py`) to reproduce the demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- 1. Main context: the RAM tier -------------------------------------------------
@dataclass
class Message:
    """One conversation turn -- lives in `MainContext.messages`, and, once
    evicted, in `MainContext.evicted`."""

    role: str
    text: str


@dataclass
class MainContext:
    """The RAM tier: a fixed-size prompt buffer always visible to the model.

    Holds two kinds of state: `core` sections that persist for the whole
    session (persona, user facts, ...) and a rolling `messages` list capped
    at `max_messages`. Appending past the cap evicts the oldest message into
    `evicted` rather than dropping it -- paged out, not deleted, so
    `conversation_search` can still find it later.

    Attributes:
        core: Persistent prompt sections, keyed by section name (e.g.
            "persona", "user"). Always rendered into the prompt.
        messages: Recent conversation turns, oldest first, capped at
            `max_messages`.
        max_messages: Maximum turns kept in `messages` before the oldest is
            evicted.
        evicted: Turns pushed out of `messages` by eviction, oldest first.
            Not lost -- still searchable via `conversation_search`.
    """

    core: dict[str, str] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 4
    evicted: list[Message] = field(default_factory=list)

    def append(self, role: str, text: str) -> None:
        """Append a turn, evicting the oldest message if over capacity.

        Args:
            role: Speaker, e.g. "user" or "assistant".
            text: Turn content.
        """
        self.messages.append(Message(role=role, text=text))
        while len(self.messages) > self.max_messages:
            self.evicted.append(self.messages.pop(0))

    def render(self) -> str:
        """Render `core` and `messages` into the prompt text the model would see.

        Returns:
            `core` sections (sorted by key) followed by `messages`, oldest
            first. `evicted` messages are intentionally left out -- they are
            no longer part of the visible context.
        """
        parts: list[str] = ["[core]"]
        for key, value in sorted(self.core.items()):
            parts.append(f"  {key}: {value}")
        parts.append("[messages]")
        for msg in self.messages:
            parts.append(f"  {msg.role}: {msg.text}")
        return "\n".join(parts)


# --- 2. Archival memory: the disk tier ----------------------------------------------
@dataclass
class ArchivalRecord:
    """One fact stored in archival memory.

    Carries provenance (`session_id`, `turn_id`) alongside the text so a
    retrieved fact can be traced back to where it came from -- without this,
    an agent can recall a fact but not say why it should be trusted.

    Attributes:
        rid: Archival record id, e.g. "a001".
        text: The stored fact.
        tags: Free-form labels for filtering/organizing (not used by
            `ArchivalStore.search` itself, which only looks at `text`).
        session_id: Session the fact was written in.
        turn_id: Turn within that session the fact was written at.
    """

    rid: str
    text: str
    tags: tuple[str, ...] = ()
    session_id: str = "s0"
    turn_id: int = 0


class ArchivalStore:
    """The disk tier: an unbounded store, paged in only via search.

    Unlike `MainContext`, nothing here is ever forced into the prompt --
    records are read when `search` is called and written when `insert` is
    called, standing in for MemGPT's `archival_memory_insert` /
    `archival_memory_search` tool pair.
    """

    def __init__(self) -> None:
        self._records: list[ArchivalRecord] = []
        self._counter = 0

    def insert(self, text: str, *, tags: tuple[str, ...] = (),
               session_id: str = "s0", turn_id: int = 0) -> str:
        """Store a fact and assign it a new record id.

        Args:
            text: The fact to store.
            tags: Optional labels for filtering/organizing.
            session_id: Session this write happened in (for provenance).
            turn_id: Turn within that session (for provenance).

        Returns:
            The new record's id, e.g. "a001".
        """
        self._counter += 1
        rid = f"a{self._counter:03d}"
        self._records.append(ArchivalRecord(
            rid=rid, text=text, tags=tags,
            session_id=session_id, turn_id=turn_id,
        ))
        return rid

    def search(self, query: str, top_k: int = 3) -> list[ArchivalRecord]:
        """Retrieve the top-k records most similar to `query` by token overlap.

        Scores each record by Jaccard similarity over lowercased whitespace
        tokens (`|query tokens ∩ text tokens| / |query tokens ∪ text tokens|`)
        -- a stand-in for the BM25/embedding retrieval a production archival
        store would use; good enough to demonstrate paging, not meant to be
        a real ranking function.

        Args:
            query: Search text.
            top_k: Maximum number of records to return.

        Returns:
            Matching records, highest-scoring first. Records with zero
            token overlap are excluded rather than ranked last.
        """
        q_tokens = set(query.lower().split())
        scored: list[tuple[float, ArchivalRecord]] = []
        for record in self._records:
            r_tokens = set(record.text.lower().split())
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens)
            if overlap == 0:
                continue
            score = overlap / (len(q_tokens) + len(r_tokens) - overlap)
            scored.append((score, record))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    def count(self) -> int:
        """Number of records currently stored."""
        return len(self._records)


# --- 3. Memory tools: the page-in/page-out interface --------------------------------
class MemoryTools:
    """The five memory tools an agent calls to read/write the two tiers.

    Each method mirrors one of MemGPT's canonical tool names 1:1, so
    `run_scripted_agent` can dispatch a `ToolCall` straight to a method of
    the same name. Every method returns a short string observation -- the
    same shape an LLM would see spliced back into its next turn after a
    real tool call (the "page fault" pattern described in the module
    docstring).
    """

    def __init__(self, main: MainContext, archival: ArchivalStore) -> None:
        self.main = main
        self.archival = archival

    def core_memory_append(self, section: str, text: str) -> str:
        """Append text to a core section, space-joined with anything already there.

        Args:
            section: Core section name, e.g. "persona" or "user".
            text: Text to append.

        Returns:
            Confirmation observation with the section's new length.
        """
        existing = self.main.core.get(section, "")
        self.main.core[section] = (existing + " " + text).strip() if existing else text
        return f"core[{section}] appended: {len(self.main.core[section])} chars"

    def core_memory_replace(self, section: str, old: str, new: str) -> str:
        """Replace one exact substring within a core section.

        Args:
            section: Core section name.
            old: Exact substring to find; the section must currently
                contain it.
            new: Replacement text.

        Returns:
            Confirmation observation, or an error string if `old` was not
            found -- returned rather than raised, since a real agent needs
            to see the failure as an observation and retry, not crash the
            loop.
        """
        current = self.main.core.get(section, "")
        if old not in current:
            return f"error: {old!r} not in core[{section}]"
        self.main.core[section] = current.replace(old, new)
        return f"core[{section}] replaced"

    def archival_memory_insert(self, text: str, tags: tuple[str, ...] = ()) -> str:
        """Write a fact to archival memory.

        Args:
            text: Fact to store.
            tags: Optional labels.

        Returns:
            Confirmation observation with the new record id and running
            count.
        """
        rid = self.archival.insert(text, tags=tags)
        return f"stored {rid} ({self.archival.count()} records)"

    def archival_memory_search(self, query: str, top_k: int = 3) -> str:
        """Search archival memory and format the hits as one observation.

        Args:
            query: Search text.
            top_k: Maximum number of hits to return.

        Returns:
            One line per hit (record id + text), or "no matches".
        """
        hits = self.archival.search(query, top_k=top_k)
        if not hits:
            return "no matches"
        return "\n".join(f"  {h.rid}: {h.text}" for h in hits)

    def conversation_search(self, query: str) -> str:
        """Substring-search past conversation turns, most recent first.

        Searches `evicted` and `messages` together, in recency order, so a
        fact that was pushed out of the visible prompt is still recoverable
        by exact text -- covers the "persistence" failure mode without
        needing a full archival write for every turn.

        Args:
            query: Substring to search for (case-insensitive).

        Returns:
            The first (most recent) matching turn, or "no matches".
        """
        q = query.lower()
        for msg in reversed(self.main.evicted + self.main.messages):
            if q in msg.text.lower():
                return f"found ({msg.role}): {msg.text}"
        return "no matches"


# --- 4. Scripted agent loop: the control flow around the tools ----------------------
@dataclass
class ToolCall:
    """One agent-issued tool call: a `MemoryTools` method name plus its kwargs."""

    name: str
    args: dict[str, Any]


def run_scripted_agent(tools: MemoryTools, script: list[ToolCall]) -> list[str]:
    """Dispatch a scripted list of tool calls against `tools`, collecting observations.

    Stands in for the LLM's decision loop: a real agent would choose which
    tool to call and with what arguments at each step, then read the
    observation before deciding the next step. Here the "decisions" are
    pre-scripted so the tier-paging behavior is deterministic and testable
    without an LLM call.

    Args:
        tools: The `MemoryTools` instance to call methods on.
        script: Tool calls to run in order.

    Returns:
        One observation string per call, same order as `script`. A call to
        an unknown tool name, or one that raises, is recorded as an
        "error: ..." string rather than stopping the run.
    """
    observations: list[str] = []
    for call in script:
        fn = getattr(tools, call.name, None)
        if fn is None:
            observations.append(f"error: unknown tool {call.name!r}")
            continue
        try:
            observations.append(fn(**call.args))
        except Exception as e:
            observations.append(f"error: {type(e).__name__}: {e}")
    return observations


# --- 5. Demo --------------------------------------------------------------------------
def main() -> None:
    """Run the two-tier memory demo end to end.

    1. Seed `MainContext` with a short conversation (`max_messages=3`).
    2. Run a scripted tool trace that writes core memory (persona, user
       facts) and archival memory (a project fact plus two reference
       facts).
    3. Append two more turns to force eviction, pushing the earliest turns
       out of `messages` into `evicted`.
    4. Page the evicted fact back in two ways: `archival_memory_search`
       (the fact was deliberately also written to archival) and
       `conversation_search` (the fact is still sitting in `evicted`,
       found by exact substring).
    """
    print("=" * 70)
    print("MEMGPT VIRTUAL CONTEXT — Phase 14, Lesson 07")
    print("=" * 70)

    main_ctx = MainContext(max_messages=3)
    archival = ArchivalStore()
    tools = MemoryTools(main_ctx, archival)

    main_ctx.append("user", "my name is ava and I ship agents for a living")
    main_ctx.append("assistant", "noted. what are you building right now?")
    main_ctx.append("user", "a retrieval bot for our sales org, 12 tools so far")
    main_ctx.append("assistant", "12 tools is in the long-horizon band; plan for drift")

    script = [
        ToolCall("core_memory_append",
                 {"section": "persona", "text": "the agent remembers user details politely"}),
        ToolCall("core_memory_append",
                 {"section": "user", "text": "name=ava, role=ships agents"}),
        ToolCall("archival_memory_insert",
                 {"text": "ava is building a retrieval bot with 12 tools for sales",
                  "tags": ("project", "ava")}),
        ToolCall("archival_memory_insert",
                 {"text": "long-horizon tool chains drift after 20 steps per BFCL V4",
                  "tags": ("bfcl", "tools")}),
        ToolCall("archival_memory_insert",
                 {"text": "sleep-time compute consolidates memory asynchronously",
                  "tags": ("letta", "memory")}),
    ]
    observations = run_scripted_agent(tools, script)

    print("\ntool trace (memory writes)")
    for call, obs in zip(script, observations):
        print(f"  {call.name}({call.args}) -> {obs}")

    print("\nfilling main context until eviction kicks in")
    main_ctx.append("user", "what were you saying about tool chains?")
    main_ctx.append("assistant", "let me check archival")

    print(f"\nmain context ({len(main_ctx.messages)} messages, "
          f"{len(main_ctx.evicted)} evicted)")
    print(main_ctx.render())

    print("\npage in: archival_memory_search('tool chains drift')")
    hit = tools.archival_memory_search("tool chains drift", top_k=2)
    print(hit)

    print("\nconversation_search for 'retrieval bot'")
    print(tools.conversation_search("retrieval bot"))

    print()
    print("pattern: memory is interrupt-driven. agent calls a tool, runtime")
    print("fetches, result splices back as observation. same as Unix read().")


if __name__ == "__main__":
    main()
