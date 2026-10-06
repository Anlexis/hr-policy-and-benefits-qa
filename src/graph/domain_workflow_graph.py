"""AgentCore Platform v1.0"""

# Inner graph — BaseGraph with a fully custom topology.
#
# It encapsulates the HR-policy retrieval pipeline:
#
#   START
#     -> input_validate   (InputValidateNode)
#     -> retrieve         (RetrieveNode)
#     -> rerank_filter    (RerankFilterNode)
#     -> generate_answer  (GenerateAnswerNode — deterministic grounded synthesis)
#     -> output_format    (OutputFormatNode)
#     -> END
#
# Called by HRPolicyQAGraphNode.get_subgraph() (graph.py). get_output() shapes
# the sub_result dict consumed by merge_output() there — the two are designed
# together and their field names must agree.
#
# Every inner node declares ANONYMOUS trust: the outer pre_process slot has
# already enforced the caller's trust level, and an inner node demanding a
# higher level would reject the passthrough context at the subgraph boundary.
# Node constructors take no arguments — the node contract is
# ``execute(self, state) -> dict``, so runtime settings reach a node through
# the initial state seeded below.

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner HR-policy retrieval workflow.

    Inherits BaseGraph directly for a fully custom node topology. initialize
    and finalize are outer backbone concerns and are not registered here.
    """

    # -- Identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return "svc_c2_001_hr_policy_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # -- Config validation ----------------------------------------------------

    def _validate_config(self) -> None:
        """No mandatory configuration.

        Every declared setting is validated by the outer graph before it is
        forwarded, and each consuming node keeps a working default, so an
        empty config is a valid one.
        """

    # -- Runtime state seeding ------------------------------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the forwarded settings and the caller's record into the inner state.

        Two things have to cross the outer -> inner boundary and neither
        crosses on its own:

        * the declared runtime settings, because the node contract is
          ``execute(self, state) -> dict`` and takes no config argument — the
          outer graph forwards them under ``config["configurable"]`` and they
          are copied into the state here, where RetrieveNode /
          RerankFilterNode / GenerateAnswerNode read them;
        * the caller's validated record, because the framework invokes a
          subgraph without forwarding the caller-data channel — the outer
          GraphNode stashes it immediately before the invoke and it is restored
          here (see src/graph/context_bridge.py).

        Dict-valued state is JSON-serialised: the checkpointer serialises state
        with msgpack, so every structured field travels as a string.
        """
        configurable = (self.config or {}).get("configurable", {}) or {}
        seeded: Dict[str, Any] = {"input_context": get_caller_input_context()}

        retrieval = configurable.get("retrieval")
        if isinstance(retrieval, dict) and retrieval:
            seeded["retrieval_config"] = to_json(retrieval)

        qa_settings = {
            key: configurable[key]
            for key in ("statutory_trigger_days", "statutory_minimum_days", "carryover_cap_days")
            if key in configurable
        }
        if qa_settings:
            seeded["qa_config"] = to_json(qa_settings)

        return seeded

    # -- Node registration ----------------------------------------------------

    def register_nodes(self) -> None:
        """Register the five domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract. Every key
        registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring ----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear retrieval pipeline."""
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing --------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph contract.

        The topology is linear, so add_conditional_edges() is not used and this
        method is never called at runtime. It returns END on error so an
        unexpected invocation cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ---------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the dict the outer graph receives as sub_result.

        Designed together with HRPolicyQAGraphNode.merge_output(): the keys
        emitted here are exactly the keys read there.
        """
        return {
            "answer": state.get("answer"),
            "citations": state.get("citations"),
            "answer_document": state.get("answer_document"),
            "reranked_docs": state.get("reranked_docs"),
            "entitlement_summary": state.get("entitlement_summary"),
            "result": state.get("result", state.get("answer_document")),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here,
            # so a run that completed without an answer would otherwise arrive
            # at the outer graph indistinguishable from one that answered.
            "error_code": state.get("error_code"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
