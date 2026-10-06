"""AgentCore Platform v1.0"""

# Backbone pre_process node — the external-facing trust and validation gate.
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level — the trust gate)
#   - Reject an empty or over-long question early (fail fast)
#   - Refuse instruction-override payloads in the question text
#   - Validate every caller-supplied field against explicit bounds and fail
#     CLOSED on any violation, naming the FIELD and never echoing the VALUE
#   - Publish the validated question and entitlement records for the pipeline
#
# Caller data arrives on two channels and both are validated by the same rules:
#
#   input          the employee's question. Either a plain-text question or a
#                  JSON envelope carrying it under question / query / text.
#   input_context  the structured channel. Fields:
#                    channel              inert slug, [a-z0-9_]{1,32}
#                    entitlement_records  list of leave-entitlement records,
#                                         entry-capped; each carries a category
#                                         label, an optional policy reference
#                                         and two finite day counts
#
# The structured channel carries the records because the question field is one
# the framework's input gate rewrites: its personal-name heuristic matches any
# two consecutive capitalised words, which is also the shape of a leave
# category ("Annual Paid Leave", "Nursing Care Leave"), so a record carried
# there would arrive with its labels replaced by a mask token and the answer
# would name the wrong entitlements. Moving the record off that field is only
# half the fix — the fields that travel on the structured channel are screened
# here instead, and a contact identifier in one of them is refused rather than
# masked.
#
# When no records are submitted the run degrades to the policy answer alone.
# There is no path on which unvalidated caller data reaches the pipeline.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
import math
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INPUT_REJECTED

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Hard cap on the question text.
_MAX_QUESTION_CHARS = 2000

# Entitlement-record entry cap applied when config/config.yaml declares none.
_DEFAULT_MAX_ENTITLEMENT_RECORDS = 20

# Day-count bounds. A calendar year cannot grant more than 366 days of leave,
# so anything above the cap is not a day count.
_DAYS_MIN = 0.0
_DAYS_MAX = 400.0

_MAX_CATEGORY_CHARS = 60
_MAX_POLICY_REFERENCE_CHARS = 80

# Caller strings that select behaviour are inert identifiers — lowercase
# alphanumerics and underscore, bounded length. Free text in such a field is
# caller-controlled output and log injection.
_SLUG_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# Entitlement labels (the category and its policy reference) are the only
# caller free text that reaches the rendered answer, so they are locked to a
# label alphabet: word characters in any script — so Japanese category names
# are accepted as readily as English ones — plus space, the section sign and
# the punctuation policy references use ("HR Policy Manual §4.1", "Childcare
# Leave (2025 revision)", "Work-Style Guidelines 3/2026"). Everything else —
# quotes, angle brackets, colons, at-signs, backslashes, braces, control
# characters and every newline — is refused, so a label can neither carry
# markup nor break the line structure of the document it is rendered into.
_LABEL_ALPHABET = r"[\w §.,()/+%#-]"
_LABEL_RE = re.compile(f"{_LABEL_ALPHABET}+")

# Control characters (except tab and newline) are stripped from the question
# before it is measured and normalised.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
# Space runs only — a label's tabs and newlines are refused, never collapsed.
_SPACES_RE = re.compile(r" +")

# Instruction-override phrasings: text addressed to the MODEL rather than an HR
# question addressed to this agent. The platform input gate refuses
# high-confidence payloads too, but this template must not depend on that —
# where the gate is absent or configured off, an unchecked payload would reach
# the retrieval path and return an answer. Refusal is stated in terms of
# BEHAVIOUR (error status, no answer, no citations), never a gate's wording.
#
# Deliberately narrow: each alternative requires the imperative override shape,
# so a genuine HR question that happens to use these words is unaffected
# ("Can I ignore the previous year's carry-over?", "Who acts as an approver
# while my manager is on leave?", "How do I forget my payroll password?" all
# pass — see the probes in tests/unit/test_caller_data_and_output_gate.py).
_INJECTION_RE = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)"
    r"|disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules)"
    r"|forget\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts)"
    r"|(?:reveal|show|print|repeat|output|disclose)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|instructions|initial\s+prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s"
    r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?(?:developer|admin|root)\s+mode"
    r"|override\s+(?:your|the)\s+(?:instruction|instructions|rules|safety)",
    re.IGNORECASE,
)

# Contact identifiers have no place in an entitlement label, and the label
# alphabet alone does not exclude them (it permits digits, spaces and hyphens).
# Only high-precision detector types are screened:
#   - the personal-name heuristics are excluded because they match any two
#     Title Case words, which is what a leave category IS ("Annual Paid Leave",
#     "Nursing Care Leave") — screening on them would refuse ordinary records;
#   - every remaining type is a shape no leave category has, and each is a
#     personal identifier that must not enter an HR answer, so a label
#     carrying one is refused outright rather than masked.
_SCREENED_PII_TYPES = frozenset({"email", "phone_jp", "phone_us", "ssn_us", "credit_card", "my_number_jp"})


def _extract_question(raw: str) -> str:
    """Pull the question out of a plain string or a JSON envelope."""
    text = raw.strip()
    if not text:
        return ""
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


def _label_error(field: str, limit: int) -> str:
    """Build the rejection message for a label field — names the field only."""
    return f"PreProcessNode: {field} must be {limit} characters or fewer of {_LABEL_ALPHABET}"


def _validate_label(value: Any, field: str, limit: int) -> Tuple[Optional[str], Optional[str]]:
    """Validate one caller label. Returns (normalised_value, error).

    Fail CLOSED: only a string of the label alphabet within the length limit is
    accepted. The rejected value is never echoed — the error names the field
    and restates the contract.

    The alphabet is checked on the value AS SUPPLIED, before space runs are
    collapsed. Normalising first would silently repair a label carrying a
    newline or a tab into an acceptable one, and caller-controlled line
    structure is exactly what this contract exists to keep out of the answer.
    """
    if not isinstance(value, str):
        return None, _label_error(field, limit)
    if not _LABEL_RE.fullmatch(value):
        return None, _label_error(field, limit)
    collapsed = _SPACES_RE.sub(" ", value).strip()
    if not collapsed or len(collapsed) > limit:
        return None, _label_error(field, limit)
    if [f for f in detect_pii(collapsed) if f.get("type") in _SCREENED_PII_TYPES]:
        return None, f"PreProcessNode: {field} must not contain contact identifiers"
    # The structured channel gets the same instruction-override screen the
    # question gets. A label is short and alphabet-restricted, but an override
    # instruction fits inside both bounds, and labels are rendered into the
    # answer — so a screen applied only to the question would leave the
    # structured channel as an open path to the same content.
    if _INJECTION_RE.search(collapsed):
        return None, f"PreProcessNode: {field} refused - instruction-override content"
    return collapsed, None


def _finite_in_range(value: Any, field: str, lo: float, hi: float) -> Tuple[Optional[float], Optional[str]]:
    """Validate one caller-supplied number. Returns (value, error).

    Fail CLOSED. Bools, strings and other non-numerics are refused, and so are
    NaN and +/-Infinity: those parse through float() and arrive intact in a raw
    JSON body, and every comparison against NaN is False — so a NaN day count
    would slide under every entitlement threshold and be reported as compliant,
    which is the one judgement this part of the answer makes. The rejected
    value is never echoed.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"PreProcessNode: {field} must be a number between {lo:g} and {hi:g}"
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None, f"PreProcessNode: {field} must be a number between {lo:g} and {hi:g}"
    return parsed, None


def _validate_records(raw: Any, max_entries: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Validate the caller's entitlement records. Returns (records, error).

    Every entry is validated in full and EVERY accepted entry is carried
    forward — the position is computed over the whole submission, not its first
    row. Errors name the field and the entry index, never the rejected value.
    """
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "PreProcessNode: entitlement_records must be a list of records"
    if len(raw) > max_entries:
        return [], f"PreProcessNode: entitlement_records must contain {max_entries} entries or fewer"

    records: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            return [], f"PreProcessNode: entitlement_records[{index}] must be a record object"

        category, error = _validate_label(
            item.get("category"), f"entitlement_records[{index}].category", _MAX_CATEGORY_CHARS
        )
        if error or category is None:
            return [], error or _label_error(f"entitlement_records[{index}].category", _MAX_CATEGORY_CHARS)

        reference = item.get("policy_reference")
        if reference is None or reference == "":
            policy_reference = ""
        else:
            policy_reference_value, error = _validate_label(
                reference,
                f"entitlement_records[{index}].policy_reference",
                _MAX_POLICY_REFERENCE_CHARS,
            )
            if error or policy_reference_value is None:
                return [], error or _label_error(
                    f"entitlement_records[{index}].policy_reference", _MAX_POLICY_REFERENCE_CHARS
                )
            policy_reference = policy_reference_value

        entitled, error = _finite_in_range(
            item.get("days_entitled"),
            f"entitlement_records[{index}].days_entitled",
            _DAYS_MIN,
            _DAYS_MAX,
        )
        if error or entitled is None:
            return [], error or f"PreProcessNode: entitlement_records[{index}].days_entitled is invalid"

        taken, error = _finite_in_range(
            item.get("days_taken"),
            f"entitlement_records[{index}].days_taken",
            _DAYS_MIN,
            _DAYS_MAX,
        )
        if error or taken is None:
            return [], error or f"PreProcessNode: entitlement_records[{index}].days_taken is invalid"

        if taken > entitled:
            return [], (
                f"PreProcessNode: entitlement_records[{index}].days_taken must not exceed "
                f"entitlement_records[{index}].days_entitled"
            )

        records.append(
            {
                "category": category,
                "policy_reference": policy_reference,
                "days_entitled": entitled,
                "days_taken": taken,
            }
        )

    return records, None


class PreProcessNode(FunctionNode):
    """Validate the incoming question and the caller's entitlement records.

    This is the outer backbone's pre_process slot and the only node declaring
    VERIFIED_EXTERNAL trust, so anonymous callers are rejected here and the
    inner domain nodes never see unauthenticated input.

    Input state keys:
        user_input:       str   — the employee's question (plain text or JSON envelope)
        input_context:    dict  — structured caller data (channel, entitlement_records)
        runtime_settings: str   — JSON settings seeded by the outer graph

    Output state keys (partial dict):
        validated_input:     str  — normalised question
        enriched_context:    str  — JSON channel metadata
        entitlement_records: str  — JSON list of validated records
        record_count:        int
        channel:             str
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _reject(self, state: AgentState, reason: str, message: str) -> Dict[str, Any]:
        """Terminal refusal: the agent will not process this content at all.

        Reserved for content a reworded retry should not get past. A request
        the caller can correct goes through _decline() instead, so a refusal is
        never presented as something a corrected request would be accepted.
        """
        logger.warning("PreProcessNode: rejected request — %s", reason)
        emit_trace_event("pre_process_validation_failed", {"reason": reason}, state)
        emit_progress(INPUT_REJECTED)
        return {"status": AgentStatus.ERROR.value, "error_log": [message]}

    def _decline(self, state: AgentState, code: str, message: str) -> Dict[str, Any]:
        """Decline a request the caller can correct, and COMPLETE the run.

        Nothing is processed and the audit event is still recorded, exactly as
        with a refusal - what differs is the reporting. Terminating here would
        end the caller's turn and surface only an exception type, leaving the
        reason reachable solely from the audit trail; completing with the
        reason lets the caller fix the value and send the request again on the
        same conversation. The message names a field, never a value.
        """
        logger.warning("PreProcessNode: declined request — %s", code)
        emit_trace_event("pre_process_validation_declined", {"reason": code}, state)
        emit_progress(INPUT_REJECTED)
        return {"status": AgentStatus.SUCCESS.value, "error_code": code, "error_log": [message]}

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context")
        settings = from_json(state.get("runtime_settings"), {}) or {}
        declared_cap = settings.get("max_entitlement_records")
        # The outer graph validates this value before seeding it; the type check
        # here keeps a hand-edited state from turning the entry cap into a
        # comparison that raises instead of refusing.
        max_records = (
            declared_cap
            if isinstance(declared_cap, int) and not isinstance(declared_cap, bool) and declared_cap > 0
            else _DEFAULT_MAX_ENTITLEMENT_RECORDS
        )

        if not isinstance(user_input, str) or not user_input.strip():
            return self._decline(state, "EMPTY_INPUT", "PreProcessNode: user_input is empty or missing")

        cleaned = _CONTROL_CHARS_RE.sub("", user_input)
        if len(cleaned) > _MAX_QUESTION_CHARS:
            return self._decline(
                state,
                "QUESTION_TOO_LONG",
                f"PreProcessNode: user_input exceeds {_MAX_QUESTION_CHARS} characters",
            )

        # Instruction-override payloads are refused here, before the envelope is
        # unwrapped, so nothing downstream ever sees them.
        if _INJECTION_RE.search(cleaned):
            return self._reject(
                state,
                "instruction_override",
                "PreProcessNode: user_input refused - instruction-override content",
            )

        question = _extract_question(cleaned)
        if not question:
            return self._decline(state, "EMPTY_INPUT", "PreProcessNode: no question found in the input payload")
        if _INJECTION_RE.search(question):
            return self._reject(
                state,
                "instruction_override",
                "PreProcessNode: user_input refused - instruction-override content",
            )

        if input_context is not None and not isinstance(input_context, dict):
            return self._decline(state, "INVALID_REQUEST", "PreProcessNode: input_context must be an object")
        context: Dict[str, Any] = input_context or {}

        channel = context.get("channel")
        if channel is None:
            channel = "unknown"
        elif not isinstance(channel, str) or not _SLUG_RE.match(channel):
            return self._decline(
                state,
                "INVALID_REQUEST",
                "PreProcessNode: input_context.channel must be a lowercase identifier "
                "(a-z, 0-9, _; 1-32 characters)",
            )

        records, error = _validate_records(context.get("entitlement_records"), max_records)
        if error:
            return self._decline(state, "INVALID_REQUEST", error)

        validated = _WS_RE.sub(" ", question).strip()

        logger.info(
            "PreProcessNode: validated question (%d chars) with %d entitlement record(s)",
            len(validated),
            len(records),
        )
        emit_trace_event(
            "pre_process_validated",
            {
                "question_length": len(validated),
                "channel": channel,
                "record_count": len(records),
                "input_format": "structured" if records else "question_only",
            },
            state,
        )

        return {
            "validated_input": validated,
            "enriched_context": to_json({"source": "HRPolicyBenefitsQAAgent", "channel": channel}),
            "entitlement_records": to_json(records),
            "record_count": len(records),
            "channel": channel,
            "status": AgentStatus.SUCCESS.value,
        }
