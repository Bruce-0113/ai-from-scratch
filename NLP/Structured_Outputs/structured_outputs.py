"""Structured outputs & constrained decoding.

Free-form LLM generation is a suggestion, not a contract: asking a model to
"return only JSON" works most of the time, and "most" is not good enough for
a parser downstream. This script walks through the three layers used to turn
"most of the time" into "always", from cheapest/least reliable to most
reliable/most involved:

1. Hand-rolled constrained decoding (`mask_logits` / `generate_constrained`):
   mask every invalid token's logit to -inf at each decoding step so the
   sampler can only ever pick a token that keeps the output inside a target
   grammar (here, a finite-state machine `fsm`). 100% valid by construction,
   at the cost of implementing/compiling the grammar yourself.
2. Outlines (`Review` + `outlines.generate.json`): the same FSM-masking idea,
   but the FSM is compiled automatically from a Pydantic schema and applied
   during generation on a local HF model.
3. Instructor (`Invoice` + `instructor.from_anthropic`): provider-agnostic —
   it does not touch logits at all, it stuffs the schema into the prompt and
   retries on Pydantic validation failure. Works with any provider, at the
   cost of extra latency/cost on retries.
4. Native vendor structured-output APIs (OpenAI `responses.create` with a
   `json_schema` response format): server-side constrained decoding, no
   local model to manage, but locks the code to that vendor.

Each block below is a standalone demo of one layer; they are not meant to be
run as a single pipeline (`sample()` in `generate_constrained` and the
per-step FSM are illustrative stand-ins, not implemented here).
"""

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
import outlines
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field



def mask_logits(logits, valid_token_ids):
    """Return a copy of `logits` with every id not in `valid_token_ids` set to -inf.

    This is the core primitive of constrained decoding: setting a token's
    logit to negative infinity makes its post-softmax probability 0, so the
    sampler can never emit it, no matter how the rest of the distribution
    looks.
    """
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    """Greedily decode `prompt` while only allowing tokens the FSM accepts.

    At each step, `fsm.valid_tokens(state, tokenizer)` computes which
    vocabulary tokens can advance the grammar without leaving an accepting
    path; those are the only logits left unmasked before sampling. The loop
    stops once the FSM reaches an accepting state, so the final decoded
    string is guaranteed to match the grammar `fsm` encodes (e.g. a JSON
    schema or regex compiled into a finite-state machine).
    """
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)


class Review(BaseModel):
    """Target schema for Outlines FSM-constrained JSON generation below."""

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


# Outlines compiles `Review`'s JSON schema into an FSM once, then reuses it
# for every generation call — every completion is guaranteed valid JSON that
# matches the schema, with zero parsing/validation step needed afterwards.
model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')


class Invoice(BaseModel):
    """Target schema for the Instructor provider-agnostic demo below."""

    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


# Instructor does not touch logits: it turns `Invoice`'s schema into prompt
# instructions and retries the call (default 3x) on Pydantic validation
# failure, so it works unmodified across any provider `instructor` wraps.
client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)


# Native vendor structured outputs: the JSON Schema is sent as part of the
# request and the constraint is enforced server-side by the provider, so
# there's no local model or grammar-compilation step at all.
client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)


