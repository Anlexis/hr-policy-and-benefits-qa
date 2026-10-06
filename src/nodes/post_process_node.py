"""AgentCore Platform v1.0"""

# Backbone post_process node — the external-output boundary for the answer
# document. Four independent layers, applied in this order:
#
#   (1) credential scan — an API key, JWT, bearer token, cloud access key,
#       connection string or credential assignment anywhere in the document
#       withholds the response entirely (sanitised stub, error status).
#       Recognition delegates to the FRAMEWORK's detect_credentials(), unioned
#       with the credential-ASSIGNMENT shape the framework does not carry, so
#       this gate can never be narrower than the scan that would otherwise
#       refuse the node and discard the withholding with it;
#   (2) grounding check — the documented contract is that policy text is quoted
#       only from passages that cleared the relevance threshold and that every
#       one of them is cited. A citation no threshold-cleared passage backs, or
#       a citation rendered when nothing cleared the threshold, withholds the
#       response. The check reads the passage list independently of the node
#       that produced the citations: if any future path ever reached the
#       renderer with an ungrounded answer, the boundary still holds;
#   (3) caller-label containment — the only caller free text this document
#       renders is the entitlement labels (each record's category and policy
#       reference). Each is re-validated at the boundary against the same
#       alphabet the input contract enforces, and a label that fails is
#       withheld wherever it appears;
#   (4) contact-identifier redaction — an email address, phone number or
#       national/payment identifier has no place in a policy answer. Any that
#       reaches the document is replaced verbatim with a fixed token, with an
#       audit event recording how many were replaced.
#
# Layer order matters. The credential scan is a PATTERN scan, so it runs BEFORE
# any token is rewritten — a rewrite landing inside a credential-shaped string
# would destroy the very pattern the scan looks for. Label containment is a
# verbatim replacement and is order-independent; the identifier redaction is a
# pattern scan and so runs after it, on text no longer holding withheld labels.
# Both pattern scans are then repeated on the rewritten document, so nothing a
# rewrite produced can escape them.
#
# The output gate is the module-level function `_security_gate_output` called
# from inside execute(), not a node instance method: the framework's own gate
# methods are final and a node-level override raises at class definition. It is
# the single copy of the recognizer — HRPolicyBenefitsQAAgent._security_gate_output
# delegates to it rather than keeping a second, driftable set.
#
# Withholding CONTAINS as well as refuses. The framework envelope resolves the
# caller-facing value as `formatted_output or result` with no regard for status,
# so _withhold() overwrites every field named in _OUTPUT_BEARING_STATE_FIELDS
# and replaces formatted_output with a TRUTHY notice — a falsy stand-in would
# hand that fallback straight back to the pre-gate document. The envelope in
# src/graph/graph.py closes the same hole from the other side, for the paths on
# which this node never returns at all.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials
from framework.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG

from src.schemas.state import from_json

logger = logging.getLogger(__name__)

# Credential recognition is delegated to the FRAMEWORK's own detector, never to
# a local pattern set. The framework scans every node result with that same
# detector and RAISES on a hit, and a raise discards the node's whole return —
# including the withholding below (see PostProcessNode._withhold). A shape the
# framework catches and this gate missed would therefore be a containment
# BYPASS, not merely a narrower gate: the framework would refuse the node, the
# clearing would never be applied, and the un-gated document would stay in
# state for the envelope to pick up. The two cannot drift apart by construction.
#
# The repo's original local set was strictly narrower than the framework's: it
# missed `sk_live_`/`sk_test_`, `AKIA…` and database connection strings
# outright, and its JWT rule required three dot-separated parts where the
# framework matches any `eyJ…` run. Those gaps are closed by delegating.
#
# One shape is kept locally because the framework does NOT carry it — a
# credential ASSIGNMENT (`password = …`). The gate is the union, so it is never
# narrower than the framework in either direction.
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Output-bearing state inventory
# ---------------------------------------------------------------------------
# Withholding a response is not a status flip. The framework envelope resolves
# the caller-facing value as `formatted_output or result` with NO regard for
# status, and src/graph/graph.py surfaces further domain fields alongside it —
# so a gate that returns ERROR while leaving those fields populated still ships
# the refused content inside the error envelope.
#
# Every State field that can carry answer text, a retrieved passage or a caller
# payload is listed here and overwritten when the gate blocks. The inert list
# holds the fields that carry no content — counts, identifiers, validated
# configuration — and may survive a violation.
#
# The two sets must together cover every field src/schemas/state.py declares:
# tests/unit/test_output_containment.py asserts the partition is TOTAL, so a
# field added to State in future cannot quietly join the survivors without a
# deliberate classification.
_OUTPUT_BEARING_STATE_FIELDS: Tuple[str, ...] = (
    "answer",
    "answer_document",
    "citations",
    "reranked_docs",
    "retrieved_docs",
    "entitlement_summary",
    "entitlement_records",
    "validated_input",
    "normalized_query",
    "result",
    "formatted_output",
)

_INERT_STATE_FIELDS: Tuple[str, ...] = (
    # A fixed internal reason code (EMPTY_INPUT / INVALID_REQUEST / ...), chosen
    # by this template and never assembled from request data, so it carries
    # nothing a containment sweep would need to withhold. The caller-facing
    # sentence it maps to is written into `result` / `formatted_output`, which
    # are output-bearing and swept as usual.
    "error_code",
    "enriched_context",
    "record_count",
    "channel",
    "runtime_settings",
    "retrieval_config",
    "qa_config",
    "trace_id",
    "correlation_id",
)

# The label alphabet the input contract enforces, kept here as the boundary's
# own copy so this layer stays independent of the pre_process node.
_LABEL_ALPHABET = r"[\w §.,()/+%#-]"
_LABEL_RE = re.compile(f"{_LABEL_ALPHABET}+")
_MAX_LABEL_CHARS = 80
_WITHHELD_LABEL = "[label withheld]"

# Identifier types redacted from the document. The personal-name heuristics are
# excluded on purpose: they match any two consecutive capitalised words, which
# is the shape of every leave category and policy title the document is made
# of, so redacting on them would blank the answer itself.
_REDACTED_PII_TYPES = frozenset({"email", "phone_jp", "phone_us", "ssn_us", "credit_card", "my_number_jp"})
_REDACTED_TOKEN = "[contact detail redacted]"

# Rendered marker of a cited source in the document.
_CITATION_MARKER = "(ref: "


def _security_gate_output(content: str) -> Optional[str]:
    """Scan the document for credential and secret material.

    Returns the violation LABEL — the finding's type name — or None when the
    document is clean. The offending value is never returned, logged or echoed:
    putting it back into this node's own result would trip the framework's
    output-side credential scan on the very return that carries the
    withholding, and a raise discards that return entirely (see _withhold).

    Recognition delegates to the framework's `detect_credentials()`, unioned
    with the credential-ASSIGNMENT shape the framework does not carry, so this
    gate is never narrower than the scan that would otherwise refuse the node.

    Module-level function, not a node instance method — the framework's gate
    methods are final and a node-level override raises at class definition.
    """
    if not content:
        return None
    findings = detect_credentials(content)
    if findings:
        return str(findings[0].get("type") or "credential")
    if _CREDENTIAL_ASSIGNMENT_RE.search(content):
        return "credential_assignment"
    return None


def _grounding_violation(document: str, state: AgentState) -> Optional[str]:
    """Check the document against the grounding contract.

    Returns a violation name, or None when the contract holds. Two ways to
    break it: citing a passage that did not clear the relevance threshold, and
    rendering a citation at all when no passage cleared it.
    """
    grounded = from_json(state.get("reranked_docs"), []) or []
    cited = from_json(state.get("citations"), []) or []
    grounded_ids = {d.get("doc_id") for d in grounded if isinstance(d, dict)}

    for citation in cited:
        if not isinstance(citation, dict) or citation.get("doc_id") not in grounded_ids:
            return "citation_not_grounded"
    if not grounded_ids and _CITATION_MARKER in document:
        return "citation_without_passage"
    return None


def _caller_labels(state: AgentState) -> List[str]:
    """Collect every caller-derived label this document renders.

    That is each entitlement record's category and its policy reference. The
    policy text, source names and headings are template or corpus content and
    are not caller-derived.
    """
    labels: List[str] = []
    summary: Dict[str, Any] = from_json(state.get("entitlement_summary"), {}) or {}
    for record in summary.get("records") or []:
        if not isinstance(record, dict):
            continue
        for field in ("category", "policy_reference"):
            value = record.get(field)
            if isinstance(value, str) and value:
                labels.append(value)
    return labels


def _contain_caller_labels(document: str, state: AgentState) -> Tuple[str, List[str]]:
    """Withhold any rendered caller label that does not satisfy the contract.

    Returns (document, withheld_count_labels). A label that is not plain text
    of the label alphabet within the length bound is replaced wherever it
    appears, so caller-controlled markup or line structure can never reach the
    external surface even if it bypassed the input contract.
    """
    withheld: List[str] = []
    contained = document
    for label in _caller_labels(state):
        if len(label) <= _MAX_LABEL_CHARS and _LABEL_RE.fullmatch(label):
            continue
        if label in contained:
            contained = contained.replace(label, _WITHHELD_LABEL)
            withheld.append(label[:16])
    return contained, withheld


def _redact_contact_identifiers(document: str) -> Tuple[str, int]:
    """Replace every screened contact identifier with a fixed token.

    Returns (document, replacement_count). Replacement runs from the end of the
    document backwards so earlier match offsets stay valid.
    """
    findings = [f for f in detect_pii(document) if f.get("type") in _REDACTED_PII_TYPES]
    if not findings:
        return document, 0
    redacted = document
    for finding in sorted(findings, key=lambda f: int(f["start"]), reverse=True):
        start, end = int(finding["start"]), int(finding["end"])
        redacted = redacted[:start] + _REDACTED_TOKEN + redacted[end:]
    return redacted, len(findings)


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Apply the output boundary and expose the final answer.

    Outer backbone post_process slot. Declared ANONYMOUS — the caller's trust
    level was enforced at the pre_process slot.

    Input state keys:
        result:              str  — the document from the inner OutputFormatNode
        answer_document:     str  — fallback when result is absent
        reranked_docs:       str  — JSON list of threshold-cleared passages
        citations:           str  — JSON list of rendered citations
        entitlement_summary: str  — JSON position carrying the caller labels

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)

    On a withheld response the partial additionally clears every field listed
    in _OUTPUT_BEARING_STATE_FIELDS — see _withhold().
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def _withhold(self, state: AgentState, event: str, reason: str, message: str) -> Dict[str, Any]:
        """Withhold the document entirely, CONTAIN it, and fail closed.

        Blocking is not a status flip. The framework envelope resolves the
        caller-facing value as ``formatted_output or result`` without consulting
        status, and src/graph/graph.py surfaces further domain fields beside it
        — so returning ERROR while leaving the assembled answer, its citations,
        the retrieved passages and the entitlement position in state would ship
        the refused content inside the error envelope. Every output-bearing
        field is therefore overwritten here, not merely left behind.

        The replacement for ``formatted_output`` is deliberately NON-EMPTY. A
        falsy stand-in ("" or {}) hands the envelope's ``or result`` fallback
        straight back to the pre-gate value — the exact leak this closes.

        ``reason`` and ``message`` name LOCATIONS and finding types only, never
        the offending value. Echoing it would put the refused string back into
        this node's own result, where the framework's output-side credential
        scan raises — and a raise DISCARDS this entire return, so the clearing
        below would never be applied at all.
        """
        logger.error("PostProcessNode: response withheld — %s", reason)
        emit_trace_event(event, {"reason": reason}, state)
        withheld = (
            f"[HR ANSWER WITHHELD: the assembled answer did not satisfy the output "
            f"contract ({reason}). Contact the HR systems team for assistance.]"
        )
        emit_progress(OUTPUT_BLOCKED)
        contained: Dict[str, Any] = {field: None for field in _OUTPUT_BEARING_STATE_FIELDS}
        contained.update(
            {
                # Truthy on purpose — see the docstring above.
                "formatted_output": withheld,
                "result": withheld,
                "status": AgentStatus.ERROR.value,
                "error_log": [message],
            }
        )
        return contained

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }
        document: str = state.get("result") or state.get("answer_document") or ""

        if not document.strip():
            logger.warning("PostProcessNode: answer document is empty — using fallback message")
            document = "[HR Policy & Benefits] No answer content generated. " "Check error_log for upstream failures."

        # (1) credential scan — pattern scan, before any rewrite
        violation = _security_gate_output(document)
        if violation:
            return self._withhold(
                state,
                "post_process_credential_violation",
                violation,
                f"PostProcessNode: credential pattern detected in the answer — {violation}",
            )

        # (2) grounding contract
        grounding = _grounding_violation(document, state)
        if grounding:
            return self._withhold(
                state,
                "post_process_grounding_violation",
                grounding,
                f"PostProcessNode: the answer failed the grounding contract — {grounding}",
            )

        # (3) caller-label containment — verbatim, order-independent
        document, withheld_labels = _contain_caller_labels(document, state)
        if withheld_labels:
            logger.warning("PostProcessNode: withheld %d caller label(s)", len(withheld_labels))
            emit_trace_event(
                "post_process_label_withheld",
                {"withheld_count": len(withheld_labels)},
                state,
            )

        # (4) contact-identifier redaction — pattern scan, after the verbatim layer
        document, redacted_count = _redact_contact_identifiers(document)
        if redacted_count:
            logger.warning("PostProcessNode: redacted %d contact identifier(s)", redacted_count)
            emit_trace_event(
                "post_process_contact_redacted",
                {"redacted_count": redacted_count},
                state,
            )

        # Re-run both pattern scans on the rewritten document: nothing a rewrite
        # produced may escape a layer that ran before it.
        violation = _security_gate_output(document)
        if violation:
            return self._withhold(
                state,
                "post_process_credential_violation",
                violation,
                f"PostProcessNode: credential pattern detected in the answer — {violation}",
            )
        if [f for f in detect_pii(document) if f.get("type") in _REDACTED_PII_TYPES]:
            return self._withhold(
                state,
                "post_process_contact_violation",
                "contact_identifier",
                "PostProcessNode: a contact identifier survived redaction in the answer",
            )

        logger.info("PostProcessNode: output boundary passed — length=%d", len(document))
        emit_trace_event("post_process_complete", {"output_length": len(document)}, state)

        return {
            "formatted_output": document,
            "result": document,
            "status": AgentStatus.SUCCESS.value,
        }
