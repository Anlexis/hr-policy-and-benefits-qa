# PB-6: invoke execution order.
#
# Verifies that the framework's node wrapper enforces, for every concrete node
# under src/nodes/: trust gate -> node_start audit -> input gate -> execute() ->
# output gate -> node_complete audit.
#
# Also verifies the full backbone invoke order for the outer agent:
#   InitializeNode -> PreProcessNode (pre_process) -> HRPolicyQAGraphNode (main)
#   -> PostProcessNode (post_process) -> FinalizeNode
#
# The invoke uses VERIFIED_EXTERNAL caller trust — the real external path. That
# context exercises the same code a deployed caller does: it clears the
# pre_process trust gate (required_trust_level = VERIFIED_EXTERNAL) and passes
# through the inner ANONYMOUS domain nodes. An internal-caller context would not
# represent a real external caller, so it is deliberately not used.
from typing import ClassVar
from framework.nodes.base_node import BaseNode
from framework.schemas.trust_level import TrustLevel

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

# ── Template-specific constants ───────────────────────────────────────────────

# Class name of the node in the `main` backbone slot.
_MAIN_SLOT_NODE = "HRPolicyQAGraphNode"

# A question that yields a grounded answer, used by the backbone invoke test.
# It grounds on the annual-paid-leave passage (HR Policy Manual §4.1) in the
# shipped policy corpus: every salient content term matches, so the passage
# clears the relevance threshold and the answer is grounded and cited rather
# than an abstention — the full happy path of the pipeline.
#
# Contract: deploy/invoke_payload.json MUST carry this exact string as its
# `input`, so the deployment's first-invoke check and this test exercise an
# identical request. test_invoke_payload_matches_pb6 below asserts that
# equality so the two can never drift.
_VALID_PAYLOAD = (
    "How many days of annual paid leave do full-time employees accrue each "
    "fiscal year, and how many days must be taken?"
)

# ─────────────────────────────────────────────────────────────────────────────


class _PrivilegedTrustGateFixture(BaseNode):
    """Always-present privileged node used to prove the S-1 negative boundary."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _security_gate_input(self, state):
        return state

    def execute(self, state):
        return {"status": "success"}

    def _security_gate_output(self, result):
        return result


def _trust_predecessor(required: TrustLevel) -> TrustLevel:
    """Return a lower valid trust level; fail loudly if the framework adds one."""
    predecessors = {
        TrustLevel.VERIFIED_EXTERNAL: TrustLevel.ANONYMOUS,
        TrustLevel.INTERNAL: TrustLevel.VERIFIED_EXTERNAL,
    }
    try:
        return predecessors[required]
    except KeyError as exc:
        raise AssertionError(f"no lower trust level defined for {required!r}") from exc


def _discover_node_classes() -> list[type]:
    """Import every module under src/nodes/ and collect concrete BaseNode subclasses."""
    from framework.nodes.base_node import BaseNode

    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError:
        return []

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


def _patch_domain_emit(monkeypatch):
    """Patch emit_trace_event in every domain node module (avoids audit-backend calls)."""
    for mod_suffix in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
    ):
        try:
            monkeypatch.setattr(
                f"src.nodes.{mod_suffix}.emit_trace_event",
                lambda *a, **k: None,
            )
        except AttributeError:
            pass  # module not yet imported / no emit symbol; fine


class TestInvokeOrder:
    """PB-6: the wrapper must run trust gate -> node_start -> input gate -> execute()
    -> output gate -> node_complete."""

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete BaseNode subclasses found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            instance = node_cls()
            # caller trust == the node's required level so the trust gate always
            # passes here; the denial branch is asserted separately in TestTrustGate.
            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            instance(state)

            expected = [
                "event:node_start",
                "security_gate_input",
                "execute",
                "security_gate_output",
                "event:node_complete",
            ]
            if order != expected:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\n" f"expected: {expected}\nactual:   {order}"
                )

        assert not failures, "\n\n".join(failures)

    def test_s1_denial_refuses_execution_before_execute(self, monkeypatch):
        """TC-08: an always-present privileged node proves the negative S-1 path."""
        import framework.nodes.base_node as base_node_module

        events: list[str] = []
        execute_calls: list[object] = []
        monkeypatch.setattr(
            base_node_module,
            "emit_trace_event",
            lambda event_type, _payload, _state: events.append(event_type),
        )
        original_execute = _PrivilegedTrustGateFixture.execute

        def spy_execute(self, state):
            execute_calls.append(state)
            return original_execute(self, state)

        monkeypatch.setattr(_PrivilegedTrustGateFixture, "execute", spy_execute)
        result = _PrivilegedTrustGateFixture()(
            {
                "caller_trust_level": _trust_predecessor(_PrivilegedTrustGateFixture.required_trust_level).value,
                "correlation_id": "tc08-s1-denial",
            }
        )

        assert result["status"] == "error"
        assert "S-1 trust gate denied" in result["error_log"][0]
        assert events == ["s1_denied"]
        assert not execute_calls


class TestTrustGate:
    """PB-6 trust gate: it runs BEFORE execute() and denies a caller whose trust
    level is below the node's required_trust_level."""

    def test_pre_process_denies_anonymous_caller(self, monkeypatch):
        """PreProcessNode (required VERIFIED_EXTERNAL) must refuse an ANONYMOUS caller."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": _VALID_PAYLOAD,
                "correlation_id": "pb6-s1-denial",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate" in e.lower() for e in result.get("error_log", [])
        ), f"expected a trust-gate denial, got error_log={result.get('error_log')}"

    def test_pre_process_admits_verified_external_caller(self, monkeypatch):
        """The same node admits a VERIFIED_EXTERNAL caller and runs execute() to SUCCESS."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": _VALID_PAYLOAD,
                "input_context": {},
                "correlation_id": "pb6-s1-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None

    def test_inner_node_denies_below_anonymous_is_noop(self, monkeypatch):
        """An inner domain node (ANONYMOUS) admits every caller — ANONYMOUS is the floor.

        This confirms the inner nodes carry the passthrough trust level (ANONYMOUS)
        so a VERIFIED_EXTERNAL outer caller flows through the subgraph boundary
        without a denial inside the domain workflow.
        """
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel
        from src.nodes.input_validate_node import InputValidateNode

        node = InputValidateNode()
        assert node.required_trust_level == TrustLevel.ANONYMOUS
        result = node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "validated_input": _VALID_PAYLOAD,
                "correlation_id": "pb6-inner-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value


class TestBackboneInvokeOrder:
    """PB-6 backbone: a full Graph().invoke() runs the 5-node backbone in order.

    Backbone order: InitializeNode -> PreProcessNode (pre_process) ->
                    HRPolicyQAGraphNode (main) ->
                    PostProcessNode (post_process) -> FinalizeNode

    Uses VERIFIED_EXTERNAL caller trust — the real external path.
    InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) is mandatory;
    NEVER use for_internal(), which would not represent a real external caller.
    """

    def _invoke(self, monkeypatch, input_context=None):
        _patch_domain_emit(monkeypatch)
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.graph import Graph, _runtime_config

        agent = Graph(config=_runtime_config())
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        return agent.invoke(_VALID_PAYLOAD, ctx=ctx, input_context=input_context or {})

    def test_backbone_invoke_succeeds_and_returns_output(self, monkeypatch):
        from framework.schemas.agent_status import AgentStatus

        result = self._invoke(monkeypatch)
        assert result.get("status") == AgentStatus.SUCCESS.value, (
            f"Expected status={AgentStatus.SUCCESS.value!r}, got: {result.get('status')!r}\n"
            f"error_log: {result.get('error_log')}"
        )
        assert result.get("output") is not None, "output must be set after a successful invoke"
        # The assembled HR Q&A document must be present in the surfaced output.
        assert "HR POLICY & BENEFITS" in result["output"]
        # ...and it must be GROUNDED — the annual-leave passage's source is cited.
        assert "HR Policy Manual" in result["output"], "the grounded answer must cite the retrieved HR policy source"

    def test_backbone_node_history_matches_expected_order(self, monkeypatch):
        result = self._invoke(monkeypatch)
        history = result.get("node_history", [])
        assert history == [
            "InitializeNode",
            "PreProcessNode",
            "HRPolicyQAGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ], f"unexpected backbone node_history: {history}"

    def test_main_slot_is_hr_policy_qa_graph_node(self):
        """The `main` backbone slot must be HRPolicyQAGraphNode (a GraphNode — Cat 2)."""
        from framework.nodes.graph_node import GraphNode
        from src.graph.graph import HRPolicyBenefitsQAAgent, HRPolicyQAGraphNode

        agent = HRPolicyBenefitsQAAgent()
        agent.compile()
        main_node = agent._nodes.get("main")
        assert main_node is not None, "main slot must be registered"
        assert isinstance(
            main_node, HRPolicyQAGraphNode
        ), f"main slot must be HRPolicyQAGraphNode, got {type(main_node).__name__}"
        assert isinstance(main_node, GraphNode), "main slot node must subclass GraphNode (Cat 2 contract)"
        assert main_node.__class__.__name__ == _MAIN_SLOT_NODE

    def test_invoke_payload_matches_pb6(self):
        """The deployment payload must carry the exact request this test asserts.

        The first-invoke check of a deployment POSTs deploy/invoke_payload.json
        as the request body, so it has to exercise the same request PB-6 asserts
        yields a grounded answer.
        """
        repo_root = Path(__file__).resolve().parents[2]
        payload_file = repo_root / "deploy" / "invoke_payload.json"
        assert payload_file.exists(), "deploy/invoke_payload.json is required to deploy"
        body = json.loads(payload_file.read_text())
        assert body.get("input") == _VALID_PAYLOAD, "deploy/invoke_payload.json['input'] must equal the PB-6 payload"
        # The payload the deployed server forwards to agent.invoke() must be a
        # non-empty question the pre_process node can extract a question from.
        from src.nodes.pre_process_node import _extract_question

        assert (
            isinstance(body["input"], str) and body["input"].strip()
        ), "invoke_payload input must be a non-empty question string"
        assert _extract_question(
            body["input"]
        ).strip(), "the pre_process node must extract a non-empty question from the payload"

    def test_invoke_payload_caller_data_satisfies_the_input_contract(self):
        """The deployment payload's caller data must pass validation unchanged.

        A payload whose records the input contract refuses would make every
        deployment's first invoke an error, so the contract is exercised here
        against the exact file that ships.
        """
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode
        from src.schemas.state import from_json

        repo_root = Path(__file__).resolve().parents[2]
        body = json.loads((repo_root / "deploy" / "invoke_payload.json").read_text())
        context = body.get("input_context") or {}
        assert context.get("entitlement_records"), "the deployment payload must exercise the caller-data channel"
        result = PreProcessNode().execute({"user_input": body["input"], "input_context": context})
        assert result["status"] == AgentStatus.SUCCESS.value, result.get("error_log")
        assert len(from_json(result["entitlement_records"])) == len(context["entitlement_records"])
        assert TrustLevel.VERIFIED_EXTERNAL is PreProcessNode.required_trust_level

    def test_backbone_invoke_computes_from_caller_records(self, monkeypatch):
        """Caller records reach the inner graph and shape the returned document."""
        result = self._invoke(
            monkeypatch,
            input_context={
                "channel": "hr_portal",
                "entitlement_records": [
                    {
                        "category": "Annual Paid Leave",
                        "policy_reference": "HR Policy Manual §4.1",
                        "days_entitled": 20,
                        "days_taken": 2,
                    }
                ],
            },
        )
        assert "Your entitlement position" in result["output"]
        assert "Annual Paid Leave" in result["output"]
        assert "remaining 18 d" in result["output"]
