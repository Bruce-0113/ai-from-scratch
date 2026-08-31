"""Context engineering pipeline for LLM applications.

Provides utilities for token budgeting, conversation history compression,
relevance-based document reordering ("lost in the middle" mitigation),
intent-based tool selection, and a top-level ContextEngine that assembles
a full prompt context under a fixed token budget.
"""

import json
from collections import OrderedDict


def count_tokens(text):
    """Estimate the token count of a text string.

    Uses a simple heuristic (word count * 1.3) rather than a real
    tokenizer, so results are approximate.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def count_tokens_json(obj):
    """Estimate the token count of a JSON-serializable object."""
    return count_tokens(json.dumps(obj))


class ContextBudget:
    """Tracks and enforces a token budget across multiple context components.

    Each component (e.g. system prompt, tools, retrieved context) is
    allocated tokens via `allocate`, truncating content as needed to stay
    within the overall available budget.
    """

    def __init__(self, max_tokens=128000, generation_reserve=4000):
        """Initialize the budget.

        Args:
            max_tokens: Total context window size in tokens.
            generation_reserve: Tokens reserved for the model's response,
                subtracted from `max_tokens` to compute `available`.
        """
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        """Allocate tokens for a named component, truncating if necessary.

        Content is truncated first to respect `max_tokens` (a per-component
        cap), then further truncated if the overall remaining budget is
        insufficient.

        Args:
            component: Name of the context component (e.g. "system_prompt").
            content: The text content to allocate space for.
            max_tokens: Optional per-component token cap.

        Returns:
            A tuple of (possibly truncated content, tokens used). Returns
            (None, 0) if no budget remains.
        """
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        """Return the number of tokens not yet allocated."""
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        """Return the fraction of the total window (including reserve) used."""
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        """Return a human-readable summary of allocations as a string."""
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"Context Budget Report ({self.max_tokens:,} token window)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'Remaining':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)


def reorder_lost_in_middle(items, scores):
    """Reorder items so the most relevant are at the start and end.

    LLMs tend to attend less to content placed in the middle of a long
    context ("lost in the middle"). This interleaves items by descending
    score into the front and back halves so high-relevance items bookend
    the sequence and low-relevance items sit in the middle.

    Args:
        items: Sequence of items to reorder.
        scores: Relevance scores parallel to `items`; higher is more relevant.

    Returns:
        A new list with items reordered by relevance.
    """
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half


def score_relevance(query, documents):
    """Score each document's relevance to a query by word overlap.

    Args:
        query: The query string.
        documents: List of document strings to score.

    Returns:
        A list of relevance scores (fraction of query words found in each
        document), parallel to `documents`.
    """
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores


class ConversationManager:
    """Manages conversation history with automatic compression.

    Recent turns are kept verbatim; once history exceeds
    `max_history_tokens`, the oldest turns are compressed into short
    summaries so the conversation context stays bounded in size.
    """

    def __init__(self, max_history_tokens=5000):
        """Initialize with an empty history.

        Args:
            max_history_tokens: Token threshold that triggers compression
                of the oldest turns into summaries.
        """
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        """Append a conversation turn and compress history if needed.

        Args:
            role: Speaker role (e.g. "user" or "assistant").
            content: The turn's text content.
        """
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        """Summarize and drop the oldest turns until under the token limit."""
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        """Produce a compact single-line summary of the given turns."""
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def get_context(self):
        """Return the full conversation context as a formatted string.

        Includes any accumulated summaries followed by the recent,
        uncompressed turns.
        """
        parts = []
        if self.summaries:
            parts.append("[Conversation Summary]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[Recent Conversation]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        """Return the estimated token count of the full context."""
        return count_tokens(self.get_context())


# Available tools with their approximate definition token cost and
# category tags used by classify_intent/select_tools.
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns in codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new calendar event",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query on the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a chart from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    """Classify a query into one or more intent categories by keyword match.

    Args:
        query: The user query string.

    Returns:
        A list of intent category names (from TOOL_REGISTRY categories)
        whose keyword score is at least half the top score. Defaults to
        ["code"] if no keywords match.
    """
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    """Select relevant tools for a query, respecting a token budget.

    Args:
        query: The user query string, used to infer intent.
        token_budget: Maximum total tokens allowed for selected tool
            definitions.

    Returns:
        A tuple of (dict of selected tool name -> tool spec, total tokens
        used by the selection).
    """
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens


class ContextEngine:
    """Top-level pipeline that assembles a full LLM context under budget.

    Combines the system prompt, selected tools, retrieved knowledge-base
    documents (reordered to mitigate lost-in-the-middle), and conversation
    history into a single ContextBudget for each query.
    """

    def __init__(self, max_tokens=128000, generation_reserve=4000):
        """Initialize the engine's budget, conversation manager, and knowledge base.

        Args:
            max_tokens: Total context window size in tokens.
            generation_reserve: Tokens reserved for the model's response.
        """
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        """Assemble a fresh, budget-constrained context for a query.

        Allocates budget in priority order: system prompt, tools,
        retrieved knowledge-base documents, conversation history, then the
        user query itself.

        Args:
            query: The incoming user query string.

        Returns:
            The ContextBudget describing the assembled context and its
            per-component token usage.
        """
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        """Process one chat turn: record it, assemble context, record the reply.

        Args:
            query: The incoming user query string.

        Returns:
            The ContextBudget produced by `assemble` for this turn.
        """
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    """Run a scripted demo of the context engineering pipeline."""
    print("=" * 60)
    print("  Context Engineering Pipeline Demo")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- Query 1: Code task ---")
    budget = engine.chat("Fix the bug in the authentication module where JWT tokens expire too early")
    print(budget.report())

    print("\n--- Query 2: Research task ---")
    budget = engine.chat("What is the best approach for implementing vector search in PostgreSQL?")
    print(budget.report())

    print("\n--- Query 3: After conversation history builds up ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"Follow-up question number {i+1} about the implementation details of the system")
        engine.conversation.add_turn("assistant", f"Here is the response to follow-up {i+1} with technical details about the architecture")

    budget = engine.chat("Now implement the changes we discussed")
    print(budget.report())

    print("\n--- Tool Selection Examples ---")
    test_queries = [
        "Fix the bug in auth.py",
        "Schedule a meeting with the team for Tuesday",
        "Show me the database query performance stats",
        "Search for best practices on error handling",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  Query: {q}")
        print(f"  Intents: {intents}")
        print(f"  Tools: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- Lost-in-the-Middle Reordering ---")
    docs = ["Doc A (most relevant)", "Doc B (somewhat relevant)", "Doc C (least relevant)",
            "Doc D (relevant)", "Doc E (moderately relevant)"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  Original order: {docs}")
    print(f"  Scores:         {scores}")
    print(f"  Reordered:      {reordered}")
    print(f"  (Most relevant at start and end, least relevant in middle)")