"""AgentCore Platform v1.0"""

# Outer graph — AgentBaseGraph with a GraphNode subclass in the `main` slot.
#
# Outer backbone (fixed 5-node skeleton — add_edges() is never overridden):
#   START -> initialize -> pre_process(PreProcessNode, VERIFIED_EXTERNAL)
#        -> main(HRPolicyQAGraphNode, GraphNode)
#        -> post_process(PostProcessNode, ANONYMOUS + output gate)
#        -> finalize -> END
#
# Inner workflow (DomainWorkflowGraph, BaseGraph):
#   START -> input_validate -> retrieve -> rerank_filter -> generate_answer
#        -> output_format -> END
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (the retrieval pipeline)
#   src/graph/context_bridge.py        <- caller-record hand-off (outer -> inner)
#
# The class name here matches the `class:` entry point in config/agent.yaml and
# the import in src/api/server.py.

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.post_process_node import _security_gate_output as _gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, to_json

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repository
# root. config/agent.yaml (the static manifest) holds only registration
# identity; every runtime parameter lives in config/config.yaml.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the file the platform registry loads and passes as
    Graph(config=...); the standalone server (src/api/server.py) reads it here
    so a deployed agent and a registry-loaded agent see identical
    configuration. Returns an empty dict — never raises — when the file is
    absent, unreadable, not valid YAML, or not a mapping (the graph then runs
    on its built-in defaults). PyYAML is imported lazily: it is a framework
    runtime dependency, so importing it on demand avoids a hard module-load
    coupling.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return cast(Dict[str, Any], loaded)


def _config_number(value: Any, lo: float, hi: float) -> Optional[float]:
    """Validate a declared numeric setting: a real number, finite, within [lo, hi].

    Bools, strings, non-numerics, NaN/Infinity and out-of-range values all
    return None, and the consumer then keeps its built-in default. Rejecting
    non-finite values matters: NaN comparisons are always False, so a NaN
    relevance threshold or entry cap would silently disable the guard it
    configures.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


def declared_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and flatten the declared runtime settings for the domain nodes.

    `config` is what the graph was constructed with — the contents of
    config/config.yaml. Every value is checked for type, finiteness and range;
    an invalid or absent key is simply not forwarded and the consuming node
    keeps its module default, so a malformed configuration file can neither
    crash graph construction nor weaken a guard.

    The `llm` block is deliberately not forwarded: answer generation is
    deterministic grounded synthesis and invokes no model (see
    docs/02_design.md), so those settings apply only to a deployment that
    swaps in a constrained model.
    """
    retrieval_raw = config.get("retrieval")
    retrieval_declared: Dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
    qa_raw = config.get("qa")
    qa_declared: Dict[str, Any] = qa_raw if isinstance(qa_raw, dict) else {}

    retrieval: Dict[str, Any] = {}
    collection = retrieval_declared.get("collection")
    if isinstance(collection, str) and collection:
        retrieval["collection"] = collection
    top_k = _config_number(retrieval_declared.get("top_k"), 1, 100)
    if top_k is not None and top_k == int(top_k):
        retrieval["top_k"] = int(top_k)
    score_threshold = _config_number(retrieval_declared.get("score_threshold"), 0.0, 1.0)
    if score_threshold is not None:
        retrieval["score_threshold"] = score_threshold
    hybrid_search = retrieval_declared.get("hybrid_search")
    if isinstance(hybrid_search, bool):
        retrieval["hybrid_search"] = hybrid_search

    declared: Dict[str, Any] = {}
    if retrieval:
        declared["retrieval"] = retrieval

    max_records = _config_number(qa_declared.get("max_entitlement_records"), 1, 1_000)
    if max_records is not None and max_records == int(max_records):
        declared["max_entitlement_records"] = int(max_records)
    for key, lo, hi in (
        ("statutory_trigger_days", 0.0, 400.0),
        ("statutory_minimum_days", 0.0, 400.0),
        ("carryover_cap_days", 0.0, 400.0),
    ):
        value = _config_number(qa_declared.get(key), lo, hi)
        if value is not None:
            declared[key] = value

    return declared


class HRPolicyQAGraphNode(GraphNode):
    """Main slot: wraps the HR policy retrieval workflow (the inner graph).

    Contracts:
      get_subgraph()  — build the inner graph, carrying the declared settings
      extract_input() — hand the inner graph the validated question, and bridge
                        the validated caller record across the boundary
      merge_output()  — map the inner get_output() dict into the outer state delta
      error_strategy  — "propagate": re-raise inner failures as SubgraphError
    """

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """Bind the declared runtime settings forwarded by the outer graph."""
        super().__init__()
        self._settings: Dict[str, Any] = dict(settings or {})

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate the inner domain workflow graph with the declared settings.

        Imported inside the method to avoid a circular import at module load
        time and to match the nested pattern.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to work on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        Completing here keeps the original reason intact.

        This override is deliberate: GraphNode.execute() is not final, and the
        marker is the only signal that distinguishes "nothing to do" from "not
        run yet".
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: Dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Hand the inner graph the question, and bridge the caller record.

        GraphNode.extract_input() returns a STRING the framework writes into
        the inner state under `user_input`. That field is one the framework's
        input gate rewrites, and its personal-name heuristic matches any two
        consecutive capitalised words — precisely the shape of a leave category
        ("Annual Paid Leave", "Nursing Care Leave"). Carrying the validated
        entitlement records there would therefore replace their labels with a
        mask token and the answer would name the wrong entitlements. Only the
        question travels on `user_input`; the records cross on the caller-data
        channel, which carries structured values through untouched.

        This is also the last hook that sees the outer state before the inner
        invoke, and the framework does not forward the caller-data channel into
        a subgraph (see src/graph/context_bridge.py), so the record is stashed
        here for the inner side to pick up.
        """
        set_caller_input_context(
            {
                "channel": state.get("channel", ""),
                "entitlement_records": state.get("entitlement_records", "[]"),
            }
        )
        return cast(str, state.get("validated_input") or state.get("user_input") or "")

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        Designed together with DomainWorkflowGraph.get_output(); returns ONLY
        changed keys — never the full state.

        citations, reranked_docs and entitlement_summary travel out with the
        document because the output boundary (PostProcessNode) re-checks the
        grounding invariant and re-validates every caller-derived label that
        reaches the rendered answer.
        """
        return {
            "answer": sub_result.get("answer"),
            "citations": sub_result.get("citations"),
            "answer_document": sub_result.get("answer_document"),
            "reranked_docs": sub_result.get("reranked_docs"),
            "entitlement_summary": sub_result.get("entitlement_summary"),
            "result": sub_result.get("result"),
            "status": sub_result.get("status"),
            # Outer reason wins. A reason already settled before the inner run
            # is the real one; the inner graph only ever sees the downstream
            # consequence of it, so taking the inner value first would replace a
            # specific reason with a generic one - and a plain sub_result.get()
            # would erase the outer reason entirely whenever the inner run did
            # not set its own.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
        }

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared runtime settings to the inner graph.

        The node contract is ``execute(self, state) -> dict``, which takes no
        config argument, so State is the only route runtime configuration takes
        to a node: this dict reaches DomainWorkflowGraph as its `config`, and
        that graph's `_extra_initial_state()` seeds the values into the inner
        state where RetrieveNode / RerankFilterNode / GenerateAnswerNode read
        them. Only keys config/config.yaml actually declares are forwarded;
        absent keys leave each node on its own default.
        """
        return {"configurable": dict(self._settings)}


class HRPolicyBenefitsQAAgent(AgentBaseGraph):
    """HR Policy & Benefits Q&A — outer graph.

    Answers an employee's question about leave, benefits, expenses and
    work-style rules strictly from the retrieved policy passages that clear the
    relevance threshold, with citations, and abstains when none do. When the
    caller also submits the employee's entitlement records, the answer states
    the position those records put them in — days entitled, taken and
    remaining per category, with the categories that need attention called out.

    Backbone: initialize -> PreProcessNode(VERIFIED_EXTERNAL)
              -> HRPolicyQAGraphNode(GraphNode)
              -> PostProcessNode(ANONYMOUS, output boundary)
              -> finalize

    Runtime configuration: the platform registry loads config/config.yaml and
    passes it as Graph(config=...); the standalone server does the same via
    _runtime_config(). AgentBaseGraph consumes max_retry from that config
    (retry routing), the validated domain settings reach the inner graph
    through HRPolicyQAGraphNode, and the validation bounds reach the
    pre_process node through the initial state — so every declared value is
    live in both deployments.

    add_edges() is NOT overridden — backbone wiring belongs to AgentBaseGraph.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "HRPolicyBenefitsQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill the 3 domain slots after super() injects initialize + finalize.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (schema_version, session_id, trust
        level) and finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        # The declared settings are validated once here and handed to the main
        # slot, so config/config.yaml is the single source for both the
        # validation bounds and the inner pipeline.
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = HRPolicyQAGraphNode(settings=declared_settings(self.config))
        self._nodes["post_process"] = PostProcessNode()

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the validated runtime settings into the outer initial state.

        The node contract takes no config argument and every node is
        constructed without one, so the settings the pre_process node needs —
        the entitlement-record entry cap — travel through State.
        """
        return {"runtime_settings": to_json(declared_settings(self.config))}

    def _security_gate_output(self, content: str) -> Optional[str]:
        """Scan outbound text for disallowed credential/secret material.

        Returns the violation label, or None when the content is clean. This
        DELEGATES to the boundary's own module-level gate rather than keeping a
        second copy of the recognizer: the docstring's claim that the runtime
        boundary "applies the identical scan" is only true if there is exactly
        one scan. Two copies drifted apart once already — the agent-class set
        and the node set were both narrower than the framework's detector — and
        a boundary narrower than the framework is a containment bypass, because
        the framework refuses the node and discards the withholding with it.
        """
        return _gate_output(content)

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Surface the answer alongside the base invoke() envelope.

        AgentBaseGraph.get_output() returns only
        ``{output, status, trace_id, correlation_id, node_history}``. A caller
        wiring this agent into an employee portal needs the answer and its
        citations as DATA rather than by parsing the rendered document, so they
        are surfaced too.

        Output-boundary invariant preserved (fail-closed):
          * ``result`` is the value the INNER workflow produced and
            merge_output() mapped out — it is pre-boundary whenever the
            boundary did not itself write it. It is surfaced ONLY on the gated
            success path. PostProcessNode does overwrite it when it withholds,
            but the node is not always the thing that ends the run: it is
            skipped entirely when the backbone routes a non-success `main`
            straight to finalize, and its return is discarded wholesale when it
            raises or when the framework's output-side credential scan refuses its result. On each
            of those paths the pre-boundary document is still sitting in state,
            so the withholding cannot be relied on here — status is.
          * the framework envelope resolves its ``output`` key as
            ``formatted_output or result`` without consulting status, so that
            fallback is re-resolved here as well: on a non-success outcome an
            absent boundary output stays absent and never becomes the raw
            inner answer.
          * ``formatted_output`` is what the gated PostProcessNode produced, so
            it is the only caller-facing value on either path (on a withheld
            response it is the boundary's own content-free notice).
          * the structured fields are surfaced ONLY when the boundary passed
            (status == SUCCESS). On any non-success outcome — including a
            withheld document — they are withheld (None).
        """
        output: Dict[str, Any] = super().get_output(state)
        # A run that completed WITHOUT carrying out the request holds the
        # sentence saying what to correct, not a domain product: none of the
        # structured fields below were produced, so none of them is released.
        # The base envelope already holds that sentence. SUCCESS here reports
        # that the run reached a defined end safely, not that the request was
        # carried out.
        if state.get("error_code"):
            return output

        succeeded = state.get("status") == AgentStatus.SUCCESS.value

        formatted_output = state.get("formatted_output")
        output["formatted_output"] = formatted_output
        if succeeded:
            output["result"] = state.get("result")
        else:
            output["result"] = None
            # Re-resolve the base envelope's value without the `or result`
            # fallback: on a non-success outcome the boundary's own output is
            # all the caller may see, and an absent one stays absent.
            output["output"] = formatted_output or None

        output["answer"] = state.get("answer") if succeeded else None
        output["citations"] = state.get("citations") if succeeded else None
        output["answer_document"] = state.get("answer_document") if succeeded else None
        output["reranked_docs"] = state.get("reranked_docs") if succeeded else None
        output["entitlement_summary"] = state.get("entitlement_summary") if succeeded else None
        return output

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Alias kept for the standalone server import; the class name matches the
# manifest's `class:` entry point.
Graph = HRPolicyBenefitsQAAgent
