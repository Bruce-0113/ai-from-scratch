"""Dialogue state tracking (DST): rule-based extraction, state updates, and LLM-driven tracking with structured output.

Covers five complementary pieces of a task-oriented DST pipeline, in the
order a system typically layers them:

1. Rule-based slot extraction (`CUISINE_SYNONYMS`, `extract_cuisine`):
   regex/keyword matching against a synonym dictionary. A strong,
   debuggable baseline for narrow domains, but brittle outside the
   canonical vocabulary it was written for.
2. Incremental state update loop (`update_state`): merges newly extracted
   slot values into the running state dict while preserving slots the
   latest utterance didn't mention, and clears any slot the user
   explicitly negated.
3. LLM-driven DST with structured output (`RestaurantState`, `llm_dst`):
   a Pydantic schema constrains the LLM's output to valid slot values,
   so the model regenerates the whole state from the dialogue history
   each turn instead of extracting deltas - this sidesteps
   append-vs-overwrite ambiguity and handles corrections for free, at
   the cost of O(n^2) total tokens across a long dialogue.
4. Joint Goal Accuracy evaluation (`joint_goal_accuracy`): the standard
   DST metric - the fraction of turns where every predicted slot exactly
   matches the gold state (all-or-nothing per turn).
5. Correction-cue detection (`CORRECTION_CUES`, `is_correction`): a
   keyword heuristic ("actually", "no wait", ...) for flagging turns that
   should overwrite a previously filled slot rather than append to it.

`update_state` (step 2) and `llm_dst` (step 3) are illustrative: `SLOT_EXTRACTORS`,
`NEGATION_CLEARS`, and `is_negated` (step 2) and `render` (step 3) are
referenced but never defined in this file - callers must supply a dict of
per-slot extractor functions (e.g. `{"cuisine": extract_cuisine, ...}`), a
list of slots eligible for negation, a negation-detection function, and a
history-to-text renderer, respectively. `instructor` is imported to
document the expected client type but isn't invoked directly here - `llm`
in `llm_dst` is assumed to already be an `instructor`-patched client
callable as `llm(prompt, response_model=...) -> RestaurantState`.
"""

from pydantic import BaseModel
from typing import Literal, Optional
import instructor

# --- 1. Rule-based slot extractor -------------------------------------------
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    """Extract a canonical cuisine name from an utterance via synonym matching.

    Case-insensitively checks `utterance` against each cuisine's synonym
    list in `CUISINE_SYNONYMS` (e.g. "pasta" or "pizza" both resolve to
    "italian"), returning the first canonical cuisine whose synonyms
    match. Brittle outside this fixed vocabulary - a dish or phrasing not
    listed in `CUISINE_SYNONYMS` will not be recognized.

    Args:
        utterance: The user's utterance text.

    Returns:
        The canonical cuisine name (a key of `CUISINE_SYNONYMS`), or None
        if no synonym matched.
    """
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None


# --- 2. State update loop ----------------------------------------------------
def update_state(state, utterance):
    """Merge slot values extracted from the latest utterance into the running state.

    Illustrative state-update loop (see module docstring: `SLOT_EXTRACTORS`,
    `NEGATION_CLEARS`, and `is_negated` are not defined in this file).
    Copies `state`, then for each configured slot extractor, overwrites
    that slot only if the extractor returns a non-None value for
    `utterance` - so slots the utterance doesn't mention are carried over
    unchanged rather than reset. Afterwards, any slot in `NEGATION_CLEARS`
    that `is_negated` flags for this utterance is cleared to None,
    applying negation after extraction so it can't be immediately
    overwritten by a same-turn extraction.

    Args:
        state: The current slot-value state, as a dict.
        utterance: The latest user utterance to extract updates from.

    Returns:
        A new state dict: `state` with extracted slots updated and
        negated slots cleared. `state` itself is not mutated.
    """
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state


# --- 3. LLM-driven DST with structured output --------------------------------
class RestaurantState(BaseModel):
    """Pydantic schema for a restaurant-booking dialogue state.

    Constrains each slot to either None (unfilled) or one of its valid
    values - closed sets for `cuisine`, `area`, and `price` via
    `Literal`, open-ended for `people` (int) and `day` (str). Passed as
    `response_model` to an `instructor`-patched LLM call (see `llm_dst`)
    so the model's output is validated against this schema instead of
    parsed as free-form JSON.
    """

    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    """Regenerate the full dialogue state from history via an instructor-constrained LLM call.

    Illustrative (see module docstring: `render` is not defined in this
    file - callers must supply a function that formats `history` as text
    for the prompt). Rather than incrementally patching individual
    slots, prompts the LLM to re-derive the entire state from the whole
    dialogue so far, which naturally handles corrections and
    append-vs-overwrite ambiguity that an incremental extractor (see
    `update_state`) has to handle with explicit rules.

    Args:
        history: The dialogue turns so far, in whatever form `render`
            expects (e.g. a list of (speaker, utterance) pairs).
        llm: An `instructor`-patched LLM client callable as
            `llm(prompt, response_model=...) -> RestaurantState`.

    Returns:
        A `RestaurantState` instance representing the updated state.
    """
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)


# --- 4. Joint Goal Accuracy evaluation ---------------------------------------
def joint_goal_accuracy(predicted_states, gold_states):
    """Compute Joint Goal Accuracy (JGA): the fraction of turns where every slot matches gold.

    Compares `predicted_states` and `gold_states` turn by turn via `==`,
    counting a turn correct only if the predicted state as a whole
    equals the gold state - a single wrong slot fails the entire turn,
    which is what makes JGA a stricter, all-or-nothing metric compared
    to per-slot accuracy.

    Args:
        predicted_states: List of predicted state objects/dicts, one per
            turn.
        gold_states: List of gold-standard state objects/dicts, one per
            turn, aligned by position with `predicted_states`.

    Returns:
        JGA as a float in [0.0, 1.0]: turns with an exact state match,
        divided by total turns.

    Raises:
        ZeroDivisionError: If `predicted_states` is empty.
    """
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)


# --- 5. Correction-cue detection ---------------------------------------------
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    """Detect whether an utterance signals a correction to a previously filled slot.

    Case-insensitively checks `utterance` for any cue phrase in
    `CORRECTION_CUES` (e.g. "actually", "no wait"). A detected
    correction should cause the caller to overwrite the relevant slot
    rather than treat the utterance as an independent addition - this
    keyword heuristic is necessarily incomplete, since many real
    corrections don't use one of these exact cue phrases.

    Args:
        utterance: The user's utterance text.

    Returns:
        True if any correction cue phrase appears in `utterance`,
        else False.
    """
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
