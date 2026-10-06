"""AgentCore Platform v1.0"""

# Inner pipeline node 4 — compose the employee-facing answer STRICTLY from the
# passages that cleared the relevance threshold, with citations, and compute
# the employee's entitlement position from the caller's validated records.
#
# Answer generation is deterministic grounded synthesis: the answer is composed
# only from the retrieved passage text, never invented, and prompts/svc_qa.j2 is
# not rendered. A deployment that wires a constrained model in renders that
# template and must hold the identical contract — no passage above the
# threshold means an explicit statement that no policy covered the question,
# never a guess.
#
# The entitlement position is arithmetic over records the pre_process node has
# already validated: days entitled, taken and remaining per category, graded
# against the statutory-uptake rule and the carry-over cap declared in
# config/config.yaml. With no records submitted the answer covers the policy
# alone.
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

from src.schemas.state import from_json, to_json
from src.services.service import summarise_entitlements

logger = logging.getLogger(__name__)

# Grading rules applied when config/config.yaml declares none. The statutory
# figures follow the employment regulations the shipped policy corpus quotes:
# once an annual entitlement reaches the trigger size, a minimum number of days
# must be taken within the year, and an untaken balance above the carry-over cap
# expires at year end.
_DEFAULT_STATUTORY_TRIGGER_DAYS = 10.0
_DEFAULT_STATUTORY_MINIMUM_DAYS = 5.0
_DEFAULT_CARRYOVER_CAP_DAYS = 40.0

_NO_COVERAGE_ANSWER = (
    "I could not find an HR policy document that covers this question with "
    "sufficient confidence. Please rephrase, or contact your HR business partner "
    "for guidance. (No policy passage met the relevance threshold.)"
)


def _grading_rules(state: AgentState) -> Dict[str, float]:
    """Read the declared grading rules from state, falling back to the defaults.

    The values were validated for type, finiteness and range by the outer graph
    before they were seeded, so a value present here is usable as-is; anything
    absent leaves the corresponding default in place.
    """
    declared = from_json(state.get("qa_config"), {}) or {}
    rules = {
        "statutory_trigger_days": _DEFAULT_STATUTORY_TRIGGER_DAYS,
        "statutory_minimum_days": _DEFAULT_STATUTORY_MINIMUM_DAYS,
        "carryover_cap_days": _DEFAULT_CARRYOVER_CAP_DAYS,
    }
    for key in rules:
        value = declared.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rules[key] = float(value)
    return rules


def _compose_grounded_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """Compose an answer grounded ONLY in the supplied passages."""
    lead = f'Based on the applicable HR policy for your question — "{query}" — here is what applies:'
    points: List[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "HR policy")
        text = (doc.get("text") or "").strip()
        points.append(f"{i}. {text}  [{source}]")
    closing = (
        "This answer is drawn directly from the cited HR policy documents. "
        "For edge cases or exceptions, confirm with your HR business partner."
    )
    return "\n\n".join([lead, "\n".join(points), closing])


class GenerateAnswerNode(FunctionNode):
    """Generate the grounded answer and the entitlement position (inner node 4).

    Inner node — ANONYMOUS trust (see module comment).

    Input state keys:
        reranked_docs:       str  — JSON list of threshold-cleared passages
        normalized_query:    str  — the employee's question
        entitlement_records: str  — JSON list of validated caller records
        qa_config:           str  — JSON grading rules seeded by the inner graph

    Output state keys (partial dict):
        answer:              str  — grounded natural-language answer
        citations:           str  — JSON list of {doc_id, source}
        entitlement_summary: str  — JSON position computed from the records
        status:              str
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

        emit_progress("Composing the answer...")
        docs: List[Dict[str, Any]] = from_json(state.get("reranked_docs"), [])
        query = state.get("normalized_query") or ""
        records: List[Dict[str, Any]] = from_json(state.get("entitlement_records"), []) or []
        rules = _grading_rules(state)

        summary = summarise_entitlements(
            records,
            statutory_trigger_days=rules["statutory_trigger_days"],
            statutory_minimum_days=rules["statutory_minimum_days"],
            carryover_cap_days=rules["carryover_cap_days"],
        )

        if not docs:
            logger.info("GenerateAnswerNode: no grounded passages — abstaining")
            emit_trace_event(
                "generate_answer_no_coverage",
                {"grounded_passages": 0, "record_count": summary["totals"]["record_count"]},
                state,
            )
            return {
                "answer": _NO_COVERAGE_ANSWER,
                "citations": to_json([]),
                "entitlement_summary": to_json(summary),
                "status": AgentStatus.SUCCESS.value,
            }

        answer = _compose_grounded_answer(query, docs)
        citations = [{"doc_id": d.get("doc_id"), "source": d.get("source")} for d in docs]

        logger.info(
            "GenerateAnswerNode: grounded_passages=%d answer_chars=%d records=%d",
            len(docs),
            len(answer),
            summary["totals"]["record_count"],
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "grounded_passages": len(docs),
                "answer_length": len(answer),
                "citation_count": len(citations),
                "record_count": summary["totals"]["record_count"],
                "entitlement_grade": summary["overall"],
            },
            state,
        )

        return {
            "answer": answer,
            "citations": to_json(citations),
            "entitlement_summary": to_json(summary),
            "status": AgentStatus.SUCCESS.value,
        }
