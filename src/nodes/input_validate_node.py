"""AgentCore Platform v1.0"""

# Inner pipeline node 1 — domain normalisation of the question, and hydration
# of the caller's record into the inner state.
#
# Distinct from the backbone pre_process node, which owns the caller contract
# (trust, bounds, screens). By the time this node runs, the question and every
# caller field have already been validated and refused if they failed; what is
# left here is domain-shaped work: unwrap a JSON envelope, collapse whitespace,
# hold the query to a sane length for retrieval, and publish the validated
# entitlement records the answer generator computes from.
#
# Inner node — ANONYMOUS trust (the outer pre_process slot already enforced the
# caller's trust level; an inner node demanding more would reject the
# passthrough context at the subgraph boundary).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Upper bound on a single question (defensive; keeps retrieval well-behaved).
_MAX_QUERY_CHARS = 2000

_WHITESPACE_RE = re.compile(r"\s+")


def _extract_question(raw: str) -> str:
    """Pull the question text out of a plain string or a JSON envelope."""
    text = raw.strip()
    if not text:
        return ""
    # Accept a JSON envelope but never require it — callers usually send plain text.
    if text[0] in "{[":
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                for key in ("question", "query", "text", "user_input"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        except (json.JSONDecodeError, ValueError):
            pass  # not JSON — treat the whole string as the question
    return text


def _hydrate_records(input_context: Any) -> List[Dict[str, Any]]:
    """Read the validated entitlement records off the caller-data channel.

    The records were validated field by field in the pre_process node and
    travel across the subgraph boundary as a JSON string. Anything else —
    including an absent channel, and including a raw caller list, which is the
    shape unvalidated data would have — yields no records, and the answer then
    covers the policy alone. This is a type check on a transport, not a second
    contract: no value is repaired here, and nothing unvalidated can arrive on
    this route.
    """
    if not isinstance(input_context, dict):
        return []
    records = from_json(input_context.get("entitlement_records"), [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


class InputValidateNode(FunctionNode):
    """Normalise the question and publish the caller's records (inner node 1).

    Input state keys:
        validated_input: str   — normalised question from the pre_process slot
                                 (falls back to user_input outside the graph)
        input_context:   dict  — the caller-data channel, seeded by the inner
                                 graph from the outer boundary

    Output state keys (partial dict):
        normalized_query:    str
        entitlement_records: str  — JSON list of validated records
        record_count:        int
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "") or ""

        question = _extract_question(raw if isinstance(raw, str) else str(raw))
        if not question:
            logger.warning("InputValidateNode: question is empty after normalisation")
            emit_trace_event("input_validate_failed", {"reason": "empty_question"}, state)
            # The caller can send a question and try again, so the run
            # completes carrying the reason instead of terminating.
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["InputValidateNode: the question is empty"],
            }

        normalized_query = _WHITESPACE_RE.sub(" ", question).strip()[:_MAX_QUERY_CHARS]
        records = _hydrate_records(state.get("input_context"))

        logger.info(
            "InputValidateNode: normalised query (%d chars), %d entitlement record(s)",
            len(normalized_query),
            len(records),
        )
        emit_trace_event(
            "input_validate_complete",
            {"query_length": len(normalized_query), "record_count": len(records)},
            state,
        )

        return {
            "normalized_query": normalized_query,
            "entitlement_records": to_json(records),
            "record_count": len(records),
            "status": AgentStatus.SUCCESS.value,
        }
