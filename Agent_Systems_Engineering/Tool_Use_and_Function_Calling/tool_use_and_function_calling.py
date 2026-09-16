"""Toy tool registry -- schema validation, argument coercion, parallel-call
dispatch, and correlation IDs. Stdlib only.

Mirrors the "Tool Use and Function Calling" lesson from rohitg00/ai-
engineering-from-scratch (phases/14-agent-engineering/06-tool-use-and-
function-calling), whose prerequisites are Phase 14.01 (Agent Loop) and
Phase 13.01 (Function Calling Deep Dive). Its framing: Toolformer (Schick
et al., NeurIPS 2023) established that tool use can be learned self-
supervised -- keep a candidate API call only if it lowers next-token loss
on the surrounding text, no human labels required. The Berkeley Function
Calling Leaderboard V4 (Patil et al., ICML 2025) is the 2026 benchmark
built on top of that: 40% agentic, 30% multi-turn, 10% live, 10% non-live,
10% hallucination. Its finding is that single-turn function calling is
essentially solved; what's left -- long-horizon chaining past ~20 steps,
dynamic tool selection, cross-turn memory, and knowing when *not* to call
a tool -- is exactly the plumbing a tool-calling runtime has to get right
before any of those failure modes are even reachable.

This module builds that plumbing, not the model side of it: a registry
that (a) presents each tool as a {name, description, input_schema} triple
in the same shape Anthropic's `input_schema` / OpenAI's
`function.parameters` expect, (b) validates and coerces whatever
arguments the model actually returned against that schema, (c) dispatches
a batch of calls issued in one turn while keeping each call's
`tool_use_id` glued to its result, and (d) turns every failure -- unknown
tool, bad schema, wrong type, an exception inside the tool -- into a
structured string observation instead of raising, so a malformed call is
something the model can read and retry against, not a crash that ends the
run.

1. ToolDef                  -- one registered tool: `name`, `description`
                                (load-bearing for which tool the model
                                picks), `input_schema`, `executor`, and a
                                `timeout_s` that is recorded but never
                                enforced -- there is no sandboxing here,
                                only the metadata a real sandbox would
                                read
2. ToolCall / ToolResult     -- one model-issued call keyed by
                                `tool_use_id`, and its outcome under that
                                same id; `dispatch` never changes this
                                id, which is the one piece of state a
                                multi-call runtime cannot let drift
3. _coerce                   -- per-field type check against the schema
                                subset (string/integer/number/boolean/
                                array/object); repairs numeric strings
                                ("4" -> 4) but rejects `bool` posing as
                                int/number and never coerces "true"/
                                "false" strings into `boolean`
4. validate                  -- one full pass over a call's arguments:
                                missing required fields, unknown fields,
                                per-field coercion, enum membership, then
                                min/max -- collects every error found
                                instead of stopping at the first
5. ToolRegistry.catalog       -- the exact list the model would see:
                                name + description + input_schema,
                                nothing executable leaks into it
6. ToolRegistry.dispatch /
   dispatch_many              -- look up the tool, validate, run the
                                executor inside a try/except, and always
                                return a `ToolResult` under the original
                                `tool_use_id`; `dispatch_many` is a plain
                                sequential loop -- "parallel" here means
                                "several tool_use blocks issued in one
                                assistant turn," per BFCL/Anthropic
                                usage, not concurrent execution
7. add / multiply / classify  -- the three toy tools `main` registers,
                                each with a description written the way
                                a real tool description should read:
                                what it does and when to prefer it
8. main                       -- registers the tools, prints the catalog
                                a model would see, then dispatches 5
                                calls in one batch: a clean call, one
                                needing string->int coercion, an enum
                                violation, a passing enum call, and a
                                call to a tool that was never registered

Run directly (`python tool_use_and_function_calling.py`) to reproduce the
demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# --- 1. Call/result data structures -----------------------------------------
@dataclass
class ToolDef:
    """One tool the registry can dispatch to.

    Attributes:
        name: Identifier the model uses to call this tool; also the
            registry key in `ToolRegistry._tools`.
        description: When/why to use this tool. Handed to the model
            verbatim via `catalog` -- the model picks a tool by reading
            this text, so a vague description causes wrong-tool-picked
            failures more often than a bad schema does.
        input_schema: JSON Schema (the subset `validate` understands:
            required, properties with type/enum/minimum/maximum) for
            this tool's arguments.
        executor: Called as `executor(**validated_args)`; its return
            value becomes the successful `ToolResult.content`.
        timeout_s: Declared per-tool timeout. Nothing in this module
            reads it -- it documents the sandboxing surface a production
            dispatcher would enforce (see the lesson's Exercise 3),
            without actually enforcing it here.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    executor: Callable[..., str]
    timeout_s: float = 5.0


@dataclass
class ToolCall:
    """One tool invocation as the model issued it.

    Attributes:
        tool_use_id: Correlation id the model attaches to this call.
            `ToolRegistry.dispatch` copies it onto the `ToolResult`
            unchanged -- swap or drop this id in a real multi-call batch
            and the wrong result gets routed back for the wrong call.
        name: Name of the tool to dispatch to.
        args: Raw arguments as the model returned them, before
            validation/coercion.
    """

    tool_use_id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of dispatching one `ToolCall`.

    Attributes:
        tool_use_id: Copied verbatim from the `ToolCall` this answers.
        ok: False for an unknown tool, a validation error, or an
            exception raised inside the executor -- `content` in every
            such case is a human-readable "... error: ..." string, not
            an exception, so a caller can feed it straight back to the
            model as the next observation.
        content: The executor's return value on success, or the error
            string on failure.
    """

    tool_use_id: str
    ok: bool
    content: str


# --- 2. Schema validation (subset: required, types, enum, min/max) ---------
def _coerce(value: Any, schema: dict[str, Any]) -> tuple[Any, str | None]:
    """Check/coerce one value against one property's schema.

    Only one kind of repair happens: a numeric string coerces into `int`
    or `float` when the target type calls for it (`"4" -> 4`). Nothing
    else is repaired -- notably a JSON `true`/`false` sent as the
    *string* `"true"`/`"false"` is rejected, not coerced, and `bool` is
    rejected as an `integer`/`number` even though Python's `bool` is an
    `int` subclass. A property with no recognized `type` (or none at
    all) always passes through unchecked.

    Returns:
        A `(value, error)` pair: `value` coerced if the type matched
        (the original `value` otherwise), and `error` either `None` or a
        human-readable reason it didn't match.
    """
    t = schema.get("type")
    if t == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            try:
                return int(value), None
            except ValueError:
                return value, f"cannot coerce string {value!r} to integer"
        return value, f"expected integer, got {type(value).__name__}"
    if t == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
        if isinstance(value, str):
            try:
                return float(value), None
            except ValueError:
                return value, f"cannot coerce string {value!r} to number"
        return value, f"expected number, got {type(value).__name__}"
    if t == "boolean":
        if isinstance(value, bool):
            return value, None
        return value, f"expected boolean, got {type(value).__name__}"
    if t == "string":
        if isinstance(value, str):
            return value, None
        return value, f"expected string, got {type(value).__name__}"
    if t == "array":
        if isinstance(value, list):
            return value, None
        return value, f"expected array, got {type(value).__name__}"
    if t == "object":
        if isinstance(value, dict):
            return value, None
        return value, f"expected object, got {type(value).__name__}"
    return value, None


def validate(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate/coerce one call's `args` against its tool's `input_schema`.

    Runs, per field: missing-required, unknown-field, `_coerce`, enum
    membership, then minimum/maximum -- each field stops at its first
    failing check (so one field reports at most one error), but
    different fields' errors all accumulate into the same list rather
    than stopping at the first bad field in the call.

    Returns:
        `(out, errors)`: `out` holds only the fields that passed every
        check (a field with any error, including a missing required
        one, is simply absent from `out` rather than present with a bad
        value); `errors` is empty iff the call was fully valid.
    """
    errors: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", [])
    out: dict[str, Any] = {}

    for name in required:
        if name not in args:
            errors.append(f"missing required: {name}")

    for name, value in args.items():
        prop = props.get(name)
        if prop is None:
            errors.append(f"unknown field: {name}")
            continue
        coerced, err = _coerce(value, prop)
        if err:
            errors.append(f"{name}: {err}")
            continue
        if "enum" in prop and coerced not in prop["enum"]:
            errors.append(f"{name}: {coerced!r} not in {prop['enum']}")
            continue
        if prop.get("type") in ("number", "integer"):
            if "minimum" in prop and coerced < prop["minimum"]:
                errors.append(f"{name}: {coerced} < minimum {prop['minimum']}")
                continue
            if "maximum" in prop and coerced > prop["maximum"]:
                errors.append(f"{name}: {coerced} > maximum {prop['maximum']}")
                continue
        out[name] = coerced

    return out, errors


# --- 3. Registry: catalog + crash-proof dispatch ----------------------------
class ToolRegistry:
    """Name -> `ToolDef` table with schema-checked, crash-proof dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """Register `tool` under `tool.name`, replacing any prior tool with that name."""
        self._tools[tool.name] = tool

    def catalog(self) -> list[dict[str, Any]]:
        """Return the {name, description, input_schema} list a model would be shown."""
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Validate and run one `ToolCall`, never raising.

        Looks up `call.name`, validates `call.args` against its schema,
        then calls the executor with the validated (coerced) arguments.
        An unknown tool name, a validation error, or an exception raised
        inside the executor all come back as `ok=False` with a
        descriptive `content` string instead of propagating -- so one
        bad call can't take down a batch of others in `dispatch_many`.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.tool_use_id, False,
                              f"error: unknown tool {call.name!r}")
        validated, errors = validate(call.args, tool.input_schema)
        if errors:
            return ToolResult(call.tool_use_id, False,
                              "validation error: " + "; ".join(errors))
        try:
            return ToolResult(call.tool_use_id, True, tool.executor(**validated))
        except Exception as e:
            return ToolResult(call.tool_use_id, False,
                              f"execution error: {type(e).__name__}: {e}")

    def dispatch_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Dispatch every call in `calls`, in order, and return their results.

        This is a plain sequential loop, not concurrent execution --
        "the model issued several tool calls in one turn" and "the
        runtime ran them at the same time" are independent facts, and
        this module only demonstrates the first one. What must not
        change between dispatching one call and dispatching many is the
        `tool_use_id` correlation, which a one-call-at-a-time loop
        already gets right.
        """
        return [self.dispatch(c) for c in calls]


# --- 4. Toy tools ------------------------------------------------------------
def add(a: int, b: int) -> str:
    """Toy executor for the `add` tool: stringified `a + b`."""
    return str(a + b)


def multiply(a: int, b: int) -> str:
    """Toy executor for the `multiply` tool: stringified `a * b`."""
    return str(a * b)


def classify(status: str) -> str:
    """Toy executor for the `classify` tool; `status` is already enum-checked by `validate`."""
    return f"classified as {status}"


# --- 5. Demo -----------------------------------------------------------------
def main() -> None:
    """Register the three toy tools, print the model-facing catalog, then
    dispatch 5 calls in one batch: a clean call, one needing string->int
    coercion, an enum violation, a passing enum call, and a call to a
    tool that was never registered -- five different observations
    `dispatch_many` returns without ever raising.
    """
    print("=" * 70)
    print("TOOL USE and FUNCTION CALLING — Phase 14, Lesson 06")
    print("=" * 70)

    reg = ToolRegistry()
    reg.register(ToolDef(
        name="add",
        description="Add two integers a and b. Use for any integer addition.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        executor=add,
    ))
    reg.register(ToolDef(
        name="multiply",
        description="Multiply two integers a and b. Prefer multiplication over looped addition.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        executor=multiply,
    ))
    reg.register(ToolDef(
        name="classify",
        description="Classify a status as one of the allowed labels.",
        input_schema={
            "type": "object",
            "properties": {"status": {"type": "string",
                                       "enum": ["open", "closed", "pending"]}},
            "required": ["status"],
        },
        executor=classify,
    ))

    print("\ncatalog (as presented to the model)")
    for entry in reg.catalog():
        print(f"  - {entry['name']}: {entry['description']}")

    calls = [
        ToolCall("u01", "add", {"a": 2, "b": 3}),
        ToolCall("u02", "multiply", {"a": "4", "b": 5}),
        ToolCall("u03", "classify", {"status": "in_progress"}),
        ToolCall("u04", "classify", {"status": "open"}),
        ToolCall("u05", "subtract", {"a": 1, "b": 2}),
    ]
    print("\nparallel dispatch (5 calls in one turn)")
    for result in reg.dispatch_many(calls):
        tag = "OK " if result.ok else "ERR"
        print(f"  {result.tool_use_id} {tag}: {result.content}")

    print()
    print("observation shape: every validation failure is a structured error")
    print("string the agent can read and retry against. never raise to the loop.")


if __name__ == "__main__":
    main()
