# Unit tests: domain nodes and graph wiring.
#
# These import the real modules and assert real behaviour: the retrieval
# relevance scorer, the threshold grounding gate, the abstention path, the
# assembled answer document, the declared trust levels, the output boundary and
# the two-layer graph composition.
#
# Audit events are patched at the node MODULE level, never via a sys.modules
# stub — that would break the real `shared` package the framework loads at
# import time. Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json


# ── Shared fixtures / helpers ─────────────────────────────────────────────────

# A question whose salient content terms fully overlap the annual-paid-leave
# passage (HR Policy Manual §4.1) in the RetrieveNode KB → score 1.0 → clears the
# 0.70 threshold → grounded, cited answer (the happy path).
GROUNDED_QUERY = (
    "How many days of annual paid leave do full-time employees accrue each "
    "fiscal year, and how many days must be taken?"
)

# A question with no HR-policy content-term overlap → retrieval returns nothing
# above the threshold → GenerateAnswerNode abstains (no hallucination).
OFFTOPIC_QUERY = "What is the capital of France and the price of Bitcoin today?"


def _passage(
    doc_id="leave-annual-001",
    source="HR Policy Manual §4.1 Annual Paid Leave",
    text="Full-time employees accrue 20 days of annual paid leave per fiscal year.",
    score=0.9,
) -> dict:
    return {"doc_id": doc_id, "source": source, "text": text, "score": score}


# -- PreProcessNode (outer pre_process, VERIFIED_EXTERNAL) --------------------


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_valid_plain_question_returns_success(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY
        ctx = from_json(result["enriched_context"])
        assert ctx["source"] == "HRPolicyBenefitsQAAgent"
        assert ctx["channel"] == "unknown"

    def test_json_envelope_question_is_extracted(self):
        payload = json.dumps({"question": "  What is the annual leave policy? "})
        result = self.node(
            {
                "user_input": payload,
                "input_context": {"channel": "portal"},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == "What is the annual leave policy?"
        assert from_json(result["enriched_context"])["channel"] == "portal"

    def test_empty_input_returns_error(self):
        result = self.node(
            {"user_input": "", "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in e.lower() for e in result["error_log"])

    def test_whitespace_only_input_returns_error(self):
        result = self.node(
            {"user_input": "   \n\t ", "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in e.lower() for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_signature_is_state_first(self):
        import inspect
        from src.nodes.pre_process_node import PreProcessNode

        params = list(inspect.signature(PreProcessNode.execute).parameters.keys())
        assert params[0] == "self" and params[1] == "state"
        assert "_invoke_impl" not in PreProcessNode.__dict__


# ── InputValidateNode (inner domain node 1, ANONYMOUS) ─────────────────────────


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_normalizes_whitespace(self):
        result = self.node(
            {"validated_input": "  How   many\n\ndays   of leave?  ", "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["normalized_query"] == "How many days of leave?"

    def test_prefers_validated_input_over_user_input(self):
        result = self.node(
            {
                "validated_input": "annual leave",
                "user_input": "childcare leave",
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["normalized_query"] == "annual leave"

    def test_falls_back_to_user_input(self):
        result = self.node({"user_input": "remote work policy", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["normalized_query"] == "remote work policy"

    def test_json_envelope_is_extracted(self):
        result = self.node(
            {
                "validated_input": json.dumps({"question": "Annual leave?"}),
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["normalized_query"] == "Annual leave?"

    def test_empty_returns_error(self):
        result = self.node({"validated_input": "", "user_input": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in e.lower() for e in result["error_log"])

    def test_length_is_bounded(self):
        long_q = "leave " * 1000  # ~6000 chars
        result = self.node({"validated_input": long_q, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert len(result["normalized_query"]) <= 2000

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RetrieveNode (inner domain node 2, ANONYMOUS) ──────────────────────────────


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_retrieves_grounded_passage_for_on_topic_query(self):
        result = self.node({"normalized_query": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_docs"])
        assert len(docs) >= 1
        top = docs[0]
        assert top["doc_id"] == "leave-annual-001"
        assert top["score"] == 1.0  # every query content-term overlaps the passage
        # scores are sorted descending
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_offtopic_query_retrieves_nothing(self):
        result = self.node({"normalized_query": OFFTOPIC_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["retrieved_docs"]) == []

    def test_empty_query_returns_error(self):
        result = self.node({"normalized_query": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e.lower() for e in result["error_log"])

    def test_top_k_config_limits_results(self):
        # "leave" matches three leave passages at score 1.0; top_k=1 keeps one.
        # The node contract takes no config argument, so settings reach the node
        # via State: the declared retrieval block is seeded into
        # state["retrieval_config"] by the inner graph.
        state = {"normalized_query": "leave", "retrieval_config": to_json({"top_k": 1})}
        result = self.node.execute(state)
        assert len(from_json(result["retrieved_docs"])) == 1

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RerankFilterNode (inner domain node 3, ANONYMOUS) ──────────────────────────


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def test_keeps_only_passages_at_or_above_threshold(self):
        docs = [
            _passage(doc_id="a", score=0.90),
            _passage(doc_id="b", score=0.50),
            _passage(doc_id="c", score=0.72),
        ]
        result = self.node({"retrieved_docs": to_json(docs), "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        kept = from_json(result["reranked_docs"])
        assert [d["doc_id"] for d in kept] == ["a", "c"]  # 0.50 dropped, sorted desc

    def test_custom_threshold_from_config(self):
        # The threshold reaches the node via state["retrieval_config"], seeded by
        # the inner graph — never as an execute() argument.
        docs = [_passage(doc_id="a", score=0.90), _passage(doc_id="c", score=0.72)]
        state = {"retrieved_docs": to_json(docs), "retrieval_config": to_json({"score_threshold": 0.80})}
        result = self.node.execute(state)
        kept = from_json(result["reranked_docs"])
        assert [d["doc_id"] for d in kept] == ["a"]

    def test_nothing_above_threshold_yields_empty(self):
        docs = [_passage(doc_id="a", score=0.10), _passage(doc_id="b", score=0.30)]
        result = self.node({"retrieved_docs": to_json(docs), "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_docs"]) == []

    def test_empty_retrieved_docs_is_safe(self):
        result = self.node({"retrieved_docs": None, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_docs"]) == []

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── GenerateAnswerNode (inner domain node 4 — LLM, grounded, ANONYMOUS) ─────────


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def test_grounded_answer_from_passage(self):
        docs = [_passage(text="Full-time employees accrue 20 days of annual paid leave.")]
        result = self.node(
            {
                "reranked_docs": to_json(docs),
                "normalized_query": GROUNDED_QUERY,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # The answer is composed ONLY from the supplied passage text (grounded).
        assert "20 days of annual paid leave" in result["answer"]
        citations = from_json(result["citations"])
        assert citations == [{"doc_id": "leave-annual-001", "source": "HR Policy Manual §4.1 Annual Paid Leave"}]

    def test_grounding_does_not_fabricate_from_other_docs(self):
        """The answer must contain only supplied-passage content — no invented material."""
        docs = [_passage(text="Full-time employees accrue 20 days of annual paid leave.")]
        answer = self.node(
            {
                "reranked_docs": to_json(docs),
                "normalized_query": GROUNDED_QUERY,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )["answer"]
        # Content that lives in OTHER KB passages must not leak into this grounded answer.
        assert "childcare" not in answer.lower()
        assert "pension" not in answer.lower()

    def test_abstains_when_no_passages(self):
        from src.nodes.generate_answer_node import _NO_COVERAGE_ANSWER

        result = self.node(
            {
                "reranked_docs": to_json([]),
                "normalized_query": OFFTOPIC_QUERY,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["answer"] == _NO_COVERAGE_ANSWER
        assert from_json(result["citations"]) == []

    def test_missing_reranked_docs_abstains(self):
        result = self.node({"normalized_query": OFFTOPIC_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "could not find" in result["answer"].lower()

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── OutputFormatNode (inner domain node 5, ANONYMOUS) ──────────────────────────


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def _state(
        self,
        answer="Based on policy, full-time staff accrue 20 days of annual paid leave.",
        citations=None,
        summary=None,
    ):
        if citations is None:
            citations = [{"doc_id": "leave-annual-001", "source": "HR Policy Manual §4.1 Annual Paid Leave"}]
        state = {
            "answer": answer,
            "citations": to_json(citations),
            "normalized_query": GROUNDED_QUERY,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        if summary is not None:
            state["entitlement_summary"] = to_json(summary)
        return state

    def test_assembles_document_with_answer_and_sources(self):
        result = self.node(self._state())
        assert result["status"] == AgentStatus.SUCCESS.value
        doc = result["answer_document"]
        assert result["result"] == doc  # backbone convention: result mirrors answer_document
        assert "HR POLICY & BENEFITS — ANSWER" in doc
        assert "20 days of annual paid leave" in doc
        assert "HR Policy Manual §4.1 Annual Paid Leave" in doc
        assert "leave-annual-001" in doc

    def test_no_citations_note_when_empty(self):
        doc = self.node(self._state(citations=[]))["answer_document"]
        assert "No policy sources cited" in doc

    def test_empty_answer_uses_fallback(self):
        result = self.node(self._state(answer=""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "HR business partner" in result["answer_document"]

    def test_renders_the_entitlement_position_from_caller_records(self):
        from src.services.service import summarise_entitlements

        summary = summarise_entitlements(
            [
                {
                    "category": "Annual Paid Leave",
                    "policy_reference": "HR Policy Manual §4.1",
                    "days_entitled": 20,
                    "days_taken": 4,
                }
            ],
            statutory_trigger_days=10,
            statutory_minimum_days=5,
            carryover_cap_days=40,
        )
        doc = self.node(self._state(summary=summary))["answer_document"]
        assert "Your entitlement position" in doc
        assert "Annual Paid Leave (HR Policy Manual §4.1)" in doc
        assert "entitled 20 d, taken 4 d, remaining 16 d" in doc
        assert "statutory uptake not yet met" in doc

    def test_omits_the_position_section_without_records(self):
        doc = self.node(self._state())["answer_document"]
        assert "Your entitlement position" not in doc

    def test_document_states_its_disclosure_schema(self):
        doc = self.node(self._state())["answer_document"]
        assert "Schema note:" in doc
        assert "computes no monetary values" in doc

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- PostProcessNode (outer post_process, the output boundary, ANONYMOUS) -----


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def test_clean_output_passes_gate(self):
        doc = "HR POLICY & BENEFITS — ANSWER\nFull-time staff accrue 20 days of annual paid leave."
        result = self.node({"result": doc, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == doc
        assert result["result"] == doc

    def test_falls_back_to_answer_document(self):
        doc = "HR POLICY & BENEFITS — ANSWER\nRemote work up to three days per week."
        result = self.node({"answer_document": doc, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == doc

    def test_empty_output_uses_fallback(self):
        result = self.node({"result": "", "answer_document": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No answer content generated" in result["formatted_output"]

    def test_credential_leak_withholds_the_document(self):
        leaky = "HR POLICY & BENEFITS — ANSWER\napi_key=sk-abcdefghij0123456789ABCDEF"
        result = self.node({"result": leaky, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert "WITHHELD" in result["formatted_output"]
        assert "sk-abcdefghij0123456789ABCDEF" not in result["formatted_output"]
        assert result["formatted_output"] == result["result"]
        assert any("credential pattern" in e for e in result["error_log"])

    def test_security_gate_output_helper_detects_and_clears(self):
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output("sk-abcdefghij0123456789ABCDEF") is not None
        assert _security_gate_output("Bearer abcdefgh12345678") is not None
        assert _security_gate_output("password = supersecret123") is not None
        assert _security_gate_output("A perfectly clean HR answer about annual leave.") is None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── Graph wiring: outer AgentBaseGraph + inner BaseGraph (Cat 2 nested) ─────────


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import HRPolicyBenefitsQAAgent, HRPolicyQAGraphNode
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = HRPolicyBenefitsQAAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], HRPolicyQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import HRPolicyBenefitsQAAgent

        agent = HRPolicyBenefitsQAAgent()
        assert agent.name == "HRPolicyBenefitsQAAgent"
        assert agent.state_schema is State

    def test_graph_alias_matches_real_class(self):
        from src.graph.graph import Graph, HRPolicyBenefitsQAAgent

        assert Graph is HRPolicyBenefitsQAAgent

    def test_main_slot_graphnode_contracts(self):
        from src.graph.graph import HRPolicyQAGraphNode

        node = HRPolicyQAGraphNode()
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        # extract_input prefers validated_input, falls back to user_input
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys(self):
        from src.graph.graph import HRPolicyQAGraphNode

        node = HRPolicyQAGraphNode()
        sub_result = {
            "answer": "A",
            "citations": "[]",
            "answer_document": "DOC",
            "reranked_docs": "[]",
            "entitlement_summary": "{}",
            "result": "DOC",
            "status": AgentStatus.SUCCESS.value,
            "node_history": ["x"],  # not forwarded by merge_output
            "correlation_id": "c",  # not forwarded by merge_output
        }
        delta = node.merge_output({}, sub_result)
        assert delta["answer_document"] == "DOC"
        assert delta["result"] == "DOC"
        assert delta["status"] == AgentStatus.SUCCESS.value
        assert set(delta.keys()) == {
            "answer",
            "citations",
            "answer_document",
            "reranked_docs",
            "entitlement_summary",
            "result",
            "status",
            # Always forwarded so a reason can never be dropped at the boundary.
            "error_code",
        }

    def test_agent_class_output_gate_detects_credentials(self):
        """The output gate is a first-class member of the agent class."""
        from src.graph.graph import HRPolicyBenefitsQAAgent

        agent = HRPolicyBenefitsQAAgent()
        assert agent._security_gate_output("sk-abcdefghij0123456789ABCDEF") is not None
        assert agent._security_gate_output("Bearer abcdefgh12345678") is not None
        assert agent._security_gate_output("A clean HR answer about leave policy.") is None


class TestInnerDomainGraph:
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

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "svc_c2_001_hr_policy_qa_workflow"
        assert g.state_schema is State

    def test_get_output_shape_matches_merge_output_contract(self):
        """Inner get_output() must emit the keys the outer merge_output() reads."""
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        out = g.get_output(
            {
                "answer": "A",
                "citations": "[]",
                "answer_document": "DOC",
                "reranked_docs": "[]",
                "status": AgentStatus.SUCCESS.value,
            }
        )
        for key in ("answer", "citations", "answer_document", "reranked_docs", "result", "status"):
            assert key in out, f"inner get_output must emit {key} for the outer merge_output"
        # result falls back to answer_document when not explicitly set
        assert out["result"] == "DOC"

    def test_inner_graph_invoke_grounded(self):
        """Standalone inner-graph invoke (ANONYMOUS caller) runs the linear RAG pipeline
        and produces a grounded, cited HR answer."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(GROUNDED_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["answer_document"] is not None
        assert "HR POLICY & BENEFITS" in result["answer_document"]
        assert result["result"] == result["answer_document"]
        # grounded → at least one passage cleared the threshold and is cited
        assert from_json(result["reranked_docs"])
        assert from_json(result["citations"])

    def test_inner_graph_invoke_abstains_off_topic(self):
        """Off-topic query → no passage clears the threshold → grounded abstention (no citations)."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(OFFTOPIC_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_docs"]) == []
        assert from_json(result["citations"]) == []
        assert "could not find" in result["answer"].lower()


# -- Trust gate (enforced by the framework's node wrapper) --------------------


class TestTrustGate:
    """The trust gate runs inside the node wrapper, BEFORE execute().

    A unit test that calls node.execute(state) directly bypasses it, so every
    node in this suite is invoked through the wrapper (node(state)) with an
    explicit caller_trust_level. These two tests assert the gate's denial and
    admission behaviour on the only node that requires elevated trust.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def test_anonymous_caller_denied_before_execute(self):
        """An ANONYMOUS caller is denied before execute() runs (fail-closed; no
        exception raised)."""
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("trust gate denied" in e.lower() for e in result["error_log"])
        # execute() never ran, so its output key is absent from the denial result
        assert "validated_input" not in result

    def test_verified_external_caller_admitted(self):
        """A VERIFIED_EXTERNAL caller clears the gate and execute() runs to SUCCESS."""
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY
