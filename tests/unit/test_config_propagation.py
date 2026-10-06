# Unit tests: runtime-configuration propagation.
#
# Proves the config-file -> inner-graph -> node path is real:
#   config/config.yaml retrieval block
#     -> declared_settings()                        (validates type, finiteness, range)
#     -> HRPolicyQAGraphNode._parent_config()       (forwards {"configurable": {...}})
#     -> DomainWorkflowGraph._extra_initial_state() (seeds state["retrieval_config"])
#     -> RetrieveNode / RerankFilterNode            (consume it on the real .invoke() path)
#
# A declared value that never reaches a node is worse than no value at all: the
# file documents a behaviour the agent does not have. These tests assert that
# changing a declared value changes retrieval behaviour end to end.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json


class TestDeclaredSettings:
    def test_runtime_config_file_is_read(self):
        """config/config.yaml is the live runtime-parameter file."""
        from src.graph.graph import _runtime_config

        config = _runtime_config()
        assert config["max_retry"] == 3
        assert config["timeout_s"] == 30
        assert config["retrieval"]["collection"] == "svc_hr_policy_and_benefi_kb"

    def test_declared_settings_validates_and_flattens(self):
        from src.graph.graph import _runtime_config, declared_settings

        declared = declared_settings(_runtime_config())
        assert declared["retrieval"] == {
            "collection": "svc_hr_policy_and_benefi_kb",
            "top_k": 5,
            "score_threshold": 0.70,
            "hybrid_search": False,
        }
        assert declared["max_entitlement_records"] == 20
        assert declared["statutory_minimum_days"] == 5

    def test_non_finite_declared_values_are_dropped(self):
        """A NaN threshold must not reach a node: every comparison against it is False."""
        from src.graph.graph import declared_settings

        declared = declared_settings(
            {
                "retrieval": {"top_k": float("nan"), "score_threshold": float("inf")},
                "qa": {"max_entitlement_records": float("nan"), "carryover_cap_days": float("-inf")},
            }
        )
        assert "top_k" not in declared.get("retrieval", {})
        assert "score_threshold" not in declared.get("retrieval", {})
        assert "max_entitlement_records" not in declared
        assert "carryover_cap_days" not in declared

    def test_out_of_range_and_wrong_typed_values_are_dropped(self):
        from src.graph.graph import declared_settings

        declared = declared_settings(
            {
                "retrieval": {"top_k": 0, "score_threshold": 2.0, "collection": 5, "hybrid_search": "yes"},
                "qa": {"statutory_minimum_days": "five"},
            }
        )
        assert declared.get("retrieval", {}) == {}
        assert "statutory_minimum_days" not in declared


class TestParentConfigForwarding:
    def test_parent_config_forwards_the_declared_settings(self):
        from src.graph.graph import HRPolicyQAGraphNode, _runtime_config, declared_settings

        settings = declared_settings(_runtime_config())
        cfg = HRPolicyQAGraphNode(settings=settings)._parent_config()
        retrieval = cfg["configurable"]["retrieval"]
        assert retrieval["collection"] == "svc_hr_policy_and_benefi_kb"
        assert retrieval["top_k"] == 5
        assert retrieval["score_threshold"] == 0.70

    def test_registered_main_slot_carries_the_declared_settings(self):
        """The graph wires the settings in: a declared value reaches the inner graph."""
        from src.graph.graph import HRPolicyBenefitsQAAgent, _runtime_config

        agent = HRPolicyBenefitsQAAgent(config=_runtime_config())
        agent.register_nodes()
        cfg = agent._nodes["main"]._parent_config()
        assert cfg.get("configurable", {}).get("retrieval"), "the declared retrieval settings must reach the main slot"

    def test_pre_process_bounds_are_seeded_into_the_initial_state(self):
        from src.graph.graph import HRPolicyBenefitsQAAgent, _runtime_config

        agent = HRPolicyBenefitsQAAgent(config=_runtime_config())
        seeded = from_json(agent._extra_initial_state()["runtime_settings"])
        assert seeded["max_entitlement_records"] == 20


class TestExtraInitialStateSeeding:
    def test_seeds_retrieval_config_into_state(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 1}}})
        seeded = g._extra_initial_state()
        assert from_json(seeded["retrieval_config"]) == {"top_k": 1}

    def test_seeds_grading_rules_into_state(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config={"configurable": {"statutory_minimum_days": 7}})
        seeded = g._extra_initial_state()
        assert from_json(seeded["qa_config"]) == {"statutory_minimum_days": 7}

    def test_seeds_only_the_caller_channel_without_config(self):
        from src.graph.context_bridge import set_caller_input_context
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        set_caller_input_context(None)
        assert DomainWorkflowGraph()._extra_initial_state() == {"input_context": {}}


class TestNodesConsumeSeededConfig:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def test_retrieve_reads_top_k_from_seeded_state(self):
        """top_k seeded into state (the real .invoke() path) limits the retrieved set."""
        from src.nodes.retrieve_node import RetrieveNode

        node = RetrieveNode()
        # "leave" matches three leave passages at score 1.0; seeded top_k=1 keeps one.
        # Invoked through __call__ so the trust gate runs; execute(self, state) is
        # the whole node contract, so the seeded state IS the settings route.
        state = {
            "normalized_query": "leave",
            "retrieval_config": to_json({"top_k": 1}),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = node(state)  # settings read from state["retrieval_config"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert len(from_json(result["retrieved_docs"])) == 1

    def test_retrieve_default_top_k_without_seed(self):
        from src.nodes.retrieve_node import RetrieveNode

        result = RetrieveNode()({"normalized_query": "leave", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        # default top_k=5 -> all three matching leave passages retrieved
        assert len(from_json(result["retrieved_docs"])) == 3

    def test_rerank_reads_threshold_from_seeded_state(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        docs = [
            {"doc_id": "a", "source": "s", "text": "t", "score": 0.90},
            {"doc_id": "c", "source": "s", "text": "t", "score": 0.72},
        ]
        state = {
            "retrieved_docs": to_json(docs),
            "retrieval_config": to_json({"score_threshold": 0.80}),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = RerankFilterNode()(state)  # config read from state["retrieval_config"]
        kept = from_json(result["reranked_docs"])
        assert [d["doc_id"] for d in kept] == ["a"]  # 0.72 dropped by the seeded 0.80 floor


class TestEndToEndManifestConfigChangesRetrieval:
    """The real .invoke() path: a declared top_k override changes the grounded result."""

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def _invoke(self, config):
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config=config)
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        return g.invoke("leave", ctx=ctx)

    def test_declared_top_k_limits_grounded_citations_end_to_end(self):
        # score_threshold 0.0 keeps every retrieved passage so the reranked count
        # equals top_k — proving the declared value drives the pipeline.
        one = self._invoke({"configurable": {"retrieval": {"top_k": 1, "score_threshold": 0.0}}})
        many = self._invoke({"configurable": {"retrieval": {"top_k": 5, "score_threshold": 0.0}}})
        assert len(from_json(one["reranked_docs"])) == 1
        assert len(from_json(many["reranked_docs"])) > 1
        assert len(from_json(one["citations"])) == 1
