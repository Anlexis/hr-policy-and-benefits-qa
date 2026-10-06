"""AgentCore Platform v1.0"""

# Inner pipeline node 5 (last) — assemble the employee-facing answer document
# from the grounded answer, the caller's entitlement position and the citation
# block. This node produces the document the backbone's output boundary then
# checks before anything is returned to the caller.
#
# Disclosure schema of the rendered document — the contract the output boundary
# enforces and the reader is told about in the document itself:
#   * policy text is quoted only from passages that cleared the relevance
#     threshold, and every one of them is cited;
#   * day counts are reported exactly as submitted in the caller's own records,
#     with the remaining balance derived from them;
#   * no monetary value is computed or rendered, so no rounding grid applies.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress

from src.schemas.state import from_json
from src.services.service import GRADE_ATTENTION, GRADE_COMPLIANT, GRADE_SHORTFALL, days_display

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

# How each grade reads in the document.
_GRADE_TEXT = {
    GRADE_SHORTFALL: "statutory uptake not yet met",
    GRADE_ATTENTION: "balance above the carry-over cap",
    GRADE_COMPLIANT: "on track",
}

_SCHEMA_NOTE = (
    "Schema note: policy text above is quoted only from the cited passages, and "
    "day counts are reported exactly as submitted in your own entitlement "
    "records. This answer computes no monetary values."
)


def _entitlement_lines(summary: Dict[str, Any]) -> List[str]:
    """Render the entitlement position, or nothing when no records were submitted."""
    records: List[Dict[str, Any]] = summary.get("records") or []
    if not records:
        return []

    lines = ["", "Your entitlement position", _SUBSEP]
    for record in records:
        reference = record.get("policy_reference") or ""
        label = f"{record.get('category', 'Entitlement')}"
        if reference:
            label = f"{label} ({reference})"
        lines.append(
            f"  {label}: entitled {days_display(record.get('days_entitled', 0))} d, "
            f"taken {days_display(record.get('days_taken', 0))} d, "
            f"remaining {days_display(record.get('days_remaining', 0))} d "
            f"- {_GRADE_TEXT.get(record.get('grade', ''), 'reviewed')}"
        )

    totals = summary.get("totals") or {}
    lines.append(
        f"  Totals: {totals.get('record_count', 0)} categories, "
        f"entitled {days_display(totals.get('days_entitled', 0))} d, "
        f"taken {days_display(totals.get('days_taken', 0))} d, "
        f"remaining {days_display(totals.get('days_remaining', 0))} d"
    )
    lines.append(f"  Overall: {_GRADE_TEXT.get(summary.get('overall', ''), 'reviewed')}")
    return lines


def _assemble_document(
    query: str,
    answer: str,
    citations: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    """Assemble the final answer document."""
    lines = [
        _SEPARATOR,
        "HR POLICY & BENEFITS — ANSWER",
        _SEPARATOR,
        "",
        f"Question: {query}" if query else "Question: (not provided)",
        "",
        "Answer",
        _SUBSEP,
        answer.strip() if answer else "(No answer generated.)",
    ]
    lines += _entitlement_lines(summary)
    lines += ["", "Sources", _SUBSEP]
    if citations:
        for i, cite in enumerate(citations, 1):
            source = cite.get("source", "HR policy")
            doc_id = cite.get("doc_id", "n/a")
            lines.append(f"  [{i}] {source}  (ref: {doc_id})")
    else:
        lines.append("  (No policy sources cited — see the answer for guidance.)")
    lines += ["", _SCHEMA_NOTE, _SEPARATOR]
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final answer document (inner node 5).

    Inner node — ANONYMOUS trust (see module comment).

    Input state keys:
        answer:              str  — grounded answer
        citations:           str  — JSON list of {doc_id, source}
        entitlement_summary: str  — JSON position computed from caller records
        normalized_query:    str  — the employee's question

    Output state keys (partial dict):
        answer_document: str
        result:          str  (same as answer_document — backbone convention)
        status:          str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without an answer, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Formatting the response...")
        answer: str = state.get("answer") or ""
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), [])
        summary: Dict[str, Any] = from_json(state.get("entitlement_summary"), {}) or {}
        query: str = state.get("normalized_query") or ""

        if not answer.strip():
            logger.warning("OutputFormatNode: answer missing — using fallback message")
            emit_trace_event("output_format_empty_answer", {"reason": "missing_answer"}, state)
            answer = "No answer content was generated. Please contact your HR business " "partner for assistance."

        document = _assemble_document(query, answer, citations, summary)

        logger.info(
            "OutputFormatNode: document_chars=%d citation_count=%d",
            len(document),
            len(citations),
        )
        emit_trace_event(
            "output_format_complete",
            {
                "document_length": len(document),
                "citation_count": len(citations),
                "record_count": (summary.get("totals") or {}).get("record_count", 0),
            },
            state,
        )

        return {
            "answer_document": document,
            "result": document,
            "status": AgentStatus.SUCCESS.value,
        }
