"""AgentCore Platform v1.0"""

# State is a flat TypedDict — never a Pydantic model. The checkpointer
# serialises state with msgpack, and a Pydantic object round-trips through it
# incorrectly, so extend AgentState with primitive-valued agent fields only and
# never place credentials or secrets in it.
#
# Every dict/list-valued field is stored as a JSON-serialised Optional[str] and
# passed through the to_json() / from_json() helpers below at each producing and
# consuming node — one contract end to end. Typing such a field as a bare
# dict/list breaks msgpack serialisation.
#
# The fields below cover both layers of the nested graph: the outer backbone
# (AgentBaseGraph) and the inner retrieval pipeline (BaseGraph).

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for the HR Policy & Benefits Q&A agent.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, ...) are inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — written by the pre_process slot
    # ------------------------------------------------------------------

    # Validated and normalised question. Raw user_input is not carried further.
    validated_input: NotRequired[Optional[str]]

    # JSON channel/request metadata: {"source": str, "channel": str}
    enriched_context: NotRequired[Optional[str]]

    # JSON list of validated entitlement records supplied by the caller. Each
    # entry: {category, policy_reference, days_entitled, days_taken}.
    entitlement_records: NotRequired[Optional[str]]

    # Number of validated entitlement records the run is working with.
    record_count: NotRequired[Optional[int]]

    # The caller's inert channel identifier, as accepted by the input contract.
    channel: NotRequired[Optional[str]]

    # JSON runtime settings validated from config/config.yaml and seeded into
    # the initial state, because the node contract takes no config argument.
    runtime_settings: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — the retrieval pipeline
    # ------------------------------------------------------------------

    # JSON retrieval settings seeded into the inner state by the inner graph:
    # {collection, top_k, score_threshold, hybrid_search}.
    retrieval_config: NotRequired[Optional[str]]

    # JSON entitlement grading rules seeded into the inner state:
    # {statutory_trigger_days, statutory_minimum_days, carryover_cap_days}.
    qa_config: NotRequired[Optional[str]]

    # Cleaned, normalised question used for retrieval.
    normalized_query: NotRequired[Optional[str]]

    # JSON list of retrieved passages; each: {doc_id, source, text, score}.
    retrieved_docs: NotRequired[Optional[str]]

    # JSON list of passages that cleared the relevance threshold, score-sorted.
    reranked_docs: NotRequired[Optional[str]]

    # Grounded answer synthesised from the threshold-cleared passages only.
    answer: NotRequired[Optional[str]]

    # JSON list of citation records; each: {doc_id, source}.
    citations: NotRequired[Optional[str]]

    # JSON entitlement position computed from the validated caller records:
    # {records: [...], totals: {...}, overall: <grade>}.
    entitlement_summary: NotRequired[Optional[str]]

    # Final assembled answer document (answer + position + citation block).
    answer_document: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — written by the post_process slot (the output boundary)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller, after the output boundary ran.
    result: NotRequired[Optional[str]]

    # Final formatted answer returned to the caller. Re-declared as
    # NotRequired[Optional[str]] for TypedDict merge compatibility — never a
    # bare type.
    formatted_output: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry. The run still completes: no retrieval is
    # performed, no answer is assembled, and the domain audit event for the
    # rejection is still emitted. Carrying this as a completion marker rather
    # than a terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the
    # caller cannot influence, are NOT reported here - those stay terminal so
    # they are not mistaken for something a reworded request would get past.
    #
    # Once set, every later domain node passes through without doing work, and
    # the value is carried across the inner/outer boundary by get_output() and
    # merge_output().
    error_code: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history inherited from AgentState
