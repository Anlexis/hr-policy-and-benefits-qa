"""AgentCore Platform v1.0"""

# Inner pipeline node 3 — rank the retrieved passages by relevance and drop any
# below the configured threshold, so only well-grounded evidence reaches the
# answer generator.
#
# Deterministic: sorts the retrieved passages by score descending and keeps
# those at or above the declared relevance threshold (default 0.70). A
# deployment may swap in a cross-encoder reranker here; the threshold is what
# keeps a weak match from being answered as if it were policy.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import math
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Default relevance floor, applied when config/config.yaml declares none.
_DEFAULT_SCORE_THRESHOLD = 0.70


class RerankFilterNode(FunctionNode):
    """Rank and threshold-filter the retrieved passages (inner node 3).

    Sorts by score descending and keeps every passage at or above the
    threshold. When nothing clears it, the passage list is empty and the answer
    generator abstains with an explicit statement that no policy passage
    covered the question.

    Inner node — ANONYMOUS trust (see module comment).

    Input state keys:
        retrieved_docs:   str  — JSON list of {doc_id, source, text, score}
        retrieval_config: str  — JSON retrieval settings seeded by the inner graph

    Output state keys (partial dict):
        reranked_docs: str  — JSON list, threshold-filtered + score-sorted
        status:        str
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

        emit_progress("Ranking the results...")
        retrieved: List[Dict[str, Any]] = from_json(state.get("retrieved_docs"), [])

        # The node contract is execute(self, state) with no config parameter,
        # so the declared threshold reaches this node through State, seeded by
        # the inner graph. Absent or malformed -> the module default.
        retrieval_cfg = from_json(state.get("retrieval_config"), {}) or {}
        declared_threshold = retrieval_cfg.get("score_threshold")
        score_threshold = (
            float(declared_threshold)
            if isinstance(declared_threshold, (int, float))
            and not isinstance(declared_threshold, bool)
            and math.isfinite(float(declared_threshold))
            else _DEFAULT_SCORE_THRESHOLD
        )

        ranked = sorted(
            (d for d in retrieved if isinstance(d, dict)),
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True,
        )
        kept = [d for d in ranked if float(d.get("score", 0.0)) >= score_threshold]

        logger.info(
            "RerankFilterNode: retrieved=%d kept=%d (score_threshold=%.2f)",
            len(retrieved),
            len(kept),
            score_threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "retrieved_count": len(retrieved),
                "kept_count": len(kept),
                "score_threshold": score_threshold,
            },
            state,
        )

        return {
            "reranked_docs": to_json(kept),
            "status": AgentStatus.SUCCESS.value,
        }
