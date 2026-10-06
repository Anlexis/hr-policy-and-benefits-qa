"""AgentCore Platform v1.0"""

# Inner pipeline node 2 — retrieve the most relevant policy passages for the
# normalised query.
#
# Retrieval is deterministic and offline: it scores a compact, in-module policy
# corpus by content-term recall, so the same question always returns the same
# passages and the template runs without an embedding service. A deployment
# that has one replaces the corpus below with a vector-store lookup driven by
# the declared retrieval settings (collection, embeddings, hybrid search) —
# everything downstream is unchanged, because the contract this node publishes
# is a scored passage list in [0, 1] and the relevance threshold applied to it
# behaves identically either way: on-topic passages clear it, off-topic ones
# fall below it and the answer generator abstains rather than inventing policy.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Set

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import RETRIEVAL_FAILED

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Defaults applied when config/config.yaml declares no retrieval block; the
# declared values are seeded into state["retrieval_config"] by the inner graph.
_DEFAULT_TOP_K = 5
_DEFAULT_COLLECTION = "svc_hr_policy_and_benefi_kb"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords (function words + policy meta-words) dropped before scoring so the
# relevance score reflects the salient HR terms in the question, not filler.
_STOPWORDS: Set[str] = set(
    "a an the of to do does did is are am was were be been being have has had "
    "and or if in on at for with within without from into by as it this that "
    "these those you your yours i we our ours they them their he she his her my "
    "me us how many much what when where which who whom why can could should "
    "would may might will shall must not no nor so than then there here about "
    "over under per each any all some more most other out up down get got give "
    "given take taken need needs want "
    # policy meta-words — present in almost every HR question, non-discriminating
    "policy policies guideline guidelines rule rules regulation regulations "
    "entitlement entitlements benefit benefits allowance allowances question "
    "questions info information detail details".split()
)

# The policy corpus this template ships with: extracts standing in for an HR
# policy manual, a benefits guide, employment regulations under Japanese labour
# law, and work-style guidelines. Replace it with your own policy set — the
# retrieval contract does not change.
_HR_POLICY_KB: List[Dict[str, str]] = [
    {
        "doc_id": "leave-annual-001",
        "source": "HR Policy Manual §4.1 Annual Paid Leave",
        "text": (
            "Full-time employees accrue 20 days of annual paid leave per fiscal year. "
            "Under the Japan Labor Standards Act, at least 5 days must be taken each year. "
            "Unused leave carries over for one year, capped at 40 days total."
        ),
    },
    {
        "doc_id": "leave-childcare-002",
        "source": "Employment Regulations §7 Childcare Leave (2025 revision)",
        "text": (
            "Childcare leave may be taken until the child reaches age two under the 2025 "
            "labor-law expansion. Both parents are eligible, and the post-birth paternity "
            "leave allows up to four weeks split into two periods."
        ),
    },
    {
        "doc_id": "benefits-health-003",
        "source": "Benefits Guide §2 Health Insurance & Pension",
        "text": (
            "All employees are enrolled in the company health insurance and the employees "
            "pension. Premiums are shared 50/50 between employee and employer and are "
            "deducted monthly from payroll."
        ),
    },
    {
        "doc_id": "expense-travel-004",
        "source": "HR Policy Manual §9 Expense & Travel Reimbursement",
        "text": (
            "Business travel expenses are reimbursed within 30 days of submitting an approved "
            "expense report with receipts. Per-diem allowances apply for overnight domestic "
            "and international travel per the travel schedule."
        ),
    },
    {
        "doc_id": "workstyle-remote-005",
        "source": "Work-Style Reform Guidelines §3 Remote & Flexible Work",
        "text": (
            "Eligible employees may work remotely up to three days per week and use flextime "
            "core hours between 10:00 and 15:00. Overtime beyond the statutory 45 hours per "
            "month requires prior manager approval under the work-style reform rules."
        ),
    },
    {
        "doc_id": "leave-special-006",
        "source": "Employment Regulations §8 Special & Nursing-Care Leave",
        "text": (
            "Special paid leave is granted for marriage, bereavement, and jury duty. Nursing-care "
            "leave of up to 93 days per eligible family member is available under the Child Care "
            "and Family Care Leave Act."
        ),
    },
]


def _content_terms(text: str) -> Set[str]:
    """Salient content terms of a string (stopwords + single chars removed)."""
    return {w for w in _TOKEN_RE.findall((text or "").lower()) if w not in _STOPWORDS and len(w) > 1}


def _score(query_terms: Set[str], doc_terms: Set[str]) -> float:
    """Content-term recall in [0, 1]: fraction of query terms present in the doc."""
    if not query_terms:
        return 0.0
    overlap = len(query_terms & doc_terms)
    if overlap == 0:
        return 0.0
    return round(overlap / len(query_terms), 4)


class RetrieveNode(FunctionNode):
    """Retrieve the most relevant policy passages for the query (inner node 2).

    Deterministic content-term-recall retrieval over the shipped policy corpus.

    Inner node — ANONYMOUS trust (see module comment).

    Input state keys:
        normalized_query: str  — cleaned query from InputValidateNode
        retrieval_config: str  — JSON retrieval settings seeded by the inner graph

    Output state keys (partial dict):
        retrieved_docs: str  — JSON-serialised list of {doc_id, source, text, score}
        status:         str
        error_log:      list[str]  (only on ERROR)
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

        emit_progress("Searching the knowledge base...")
        query = state.get("normalized_query") or ""

        # The node contract is execute(self, state) with no config parameter,
        # so runtime settings reach this node through State: the outer graph
        # validates config/config.yaml once and the inner graph seeds the
        # declared retrieval block into state["retrieval_config"]. Absent -> the
        # module defaults above.
        retrieval_cfg = from_json(state.get("retrieval_config"), {}) or {}
        declared_top_k = retrieval_cfg.get("top_k")
        top_k = (
            int(declared_top_k)
            if isinstance(declared_top_k, int) and not isinstance(declared_top_k, bool) and declared_top_k > 0
            else _DEFAULT_TOP_K
        )
        declared_collection = retrieval_cfg.get("collection")
        collection = (
            declared_collection if isinstance(declared_collection, str) and declared_collection else _DEFAULT_COLLECTION
        )

        if not query.strip():
            logger.warning("RetrieveNode: normalized_query missing — nothing to retrieve")
            emit_trace_event(
                "retrieve_failed",
                {"reason": "empty_query", "collection": collection},
                state,
            )
            emit_progress(RETRIEVAL_FAILED)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: normalized_query is empty"],
            }

        query_terms = _content_terms(query)
        scored: List[Dict[str, Any]] = []
        for doc in _HR_POLICY_KB:
            doc_terms = _content_terms(doc["text"] + " " + doc["source"])
            score = _score(query_terms, doc_terms)
            if score > 0.0:
                scored.append(
                    {
                        "doc_id": doc["doc_id"],
                        "source": doc["source"],
                        "text": doc["text"],
                        "score": score,
                    }
                )

        scored.sort(key=lambda d: d["score"], reverse=True)
        retrieved = scored[:top_k]

        logger.info(
            "RetrieveNode: collection=%s query_terms=%d retrieved=%d (top_k=%d)",
            collection,
            len(query_terms),
            len(retrieved),
            top_k,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "collection": collection,
                "top_k": top_k,
                "retrieved_count": len(retrieved),
            },
            state,
        )

        return {
            "retrieved_docs": to_json(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
