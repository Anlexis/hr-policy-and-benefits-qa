# SVC-C2-001 — Output-gate CONTAINMENT (both halves).
#
# Withholding a response is not a status flip. Two independent mechanisms would
# each, on their own, hand the caller the answer the boundary refused:
#
#   (a) AgentBaseGraph.get_output() resolves its `output` key as
#       `formatted_output or result` WITHOUT consulting status, and `result` is
#       the value merge_output() mapped out of the inner workflow — pre-boundary
#       whenever PostProcessNode did not itself write it;
#   (b) PostProcessNode is not always the thing that ends the run. The backbone
#       routes a non-success `main` straight to finalize (the node never runs),
#       and BaseNode.__call__ replaces a raising node's return with a bare ERROR
#       partial that clears nothing — so the withholding is discarded while the
#       pre-boundary document stays in state.
#
# So the gate clears the state, AND the envelope withholds on status. Each is
# pinned by a test that the OTHER guard cannot rescue: the node-level
# assertions below read the node's own returned partial (the envelope is not
# involved), and the envelope assertions call get_output() on a state dict
# directly (no node is involved). tests/proof_of_boundary/test_invoke_e2e.py
# pins the end-to-end consequence on the real /invoke surface.

import ast
import json
import pathlib

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials

from src.graph.graph import HRPolicyBenefitsQAAgent
from src.nodes.post_process_node import (
    _INERT_STATE_FIELDS,
    _OUTPUT_BEARING_STATE_FIELDS,
    PostProcessNode,
    _security_gate_output,
)
from src.schemas.state import to_json

# What the inner workflow produced BEFORE the boundary ran. Every string here is
# a released value that a withheld response must not carry.
_DOCUMENT = (
    "HR POLICY & BENEFITS — ANSWER\n"
    "Full-time employees accrue 20 days of annual paid leave.  (ref: leave-annual-001)\n"
    "Annual Paid Leave (HR Policy Manual §4.1): entitled 20 d, taken 3 d, remaining 17 d"
)
_RELEASED = (
    "Full-time employees accrue 20 days",
    "leave-annual-001",
    "entitled 20 d",
    "HR Policy Manual",
)


def _state(**overrides):
    state = {
        "result": _DOCUMENT,
        "answer_document": _DOCUMENT,
        "answer": "Full-time employees accrue 20 days of annual paid leave.",
        "citations": to_json([{"doc_id": "leave-annual-001", "source": "HR Policy Manual §4.1"}]),
        "reranked_docs": to_json([]),
        "entitlement_summary": to_json(
            {"records": [{"category": "Annual Paid Leave", "policy_reference": "HR Policy Manual §4.1"}]}
        ),
        "entitlement_records": to_json([{"category": "Annual Paid Leave", "days_entitled": 20}]),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _silence_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)


def _blocked():
    """Drive the FULL node path (so the framework's output-side credential scan really runs) into a
    withheld response, using the shipped grounding branch: a citation rendered
    with nothing that cleared the relevance threshold."""
    return PostProcessNode()(_state())


# ---------------------------------------------------------------------------
# Half 1 — the gate clears the state
# ---------------------------------------------------------------------------


class TestWithholdContainsTheDocument:
    def test_the_block_actually_fires(self):
        result = _blocked()
        assert result["status"] == AgentStatus.ERROR.value
        assert "citation_not_grounded" in result["error_log"][0]

    def test_every_output_bearing_field_is_overwritten(self):
        result = _blocked()
        for field in _OUTPUT_BEARING_STATE_FIELDS:
            if field in ("formatted_output", "result"):
                continue  # replaced by the notice, asserted below
            assert result[field] is None, f"{field} still carries the pre-boundary value"

    def test_no_released_string_survives_the_block(self):
        blob = json.dumps(_blocked(), ensure_ascii=False)
        for needle in _RELEASED:
            assert needle not in blob, f"released string {needle!r} survived the block"

    def test_the_replacement_defeats_the_envelope_fallback(self):
        """`formatted_output or result` must resolve to the withholding notice.

        A falsy replacement ("" or {}) is exactly the hole this closes: it hands
        the resolution straight back to the pre-boundary document.
        """
        result = _blocked()
        assert result["formatted_output"], "the replacement must be TRUTHY — see the fallback"
        resolved = result["formatted_output"] or result["result"]
        assert resolved is result["formatted_output"]
        assert "WITHHELD" in resolved

    def test_the_violation_names_a_location_never_a_value(self):
        """Echoing the offending value would put it back into this node's own
        result, where the framework's output-side credential scan raises — and a raise DISCARDS
        the whole return, so the clearing above would never be applied."""
        leak = "AKIA" + "A" * 16
        result = PostProcessNode()(
            _state(result=f"{_DOCUMENT}\n{leak}", reranked_docs=to_json([]), citations=to_json([]))
        )
        blob = json.dumps(result, ensure_ascii=False)
        assert result["status"] == AgentStatus.ERROR.value
        assert leak not in blob
        assert "Traceback" not in blob, "the framework credential scan discarded the clearing return"
        assert "src/nodes/" not in blob
        assert result["formatted_output"], "the clearing return did not survive"


class TestOutputBearingInventoryIsTotal:
    """A field added to State in future cannot quietly join the survivors.

    The gate clears a NAMED set, so an unclassified new field would silently
    become a survivor of every violation. This forces the classification.
    """

    @staticmethod
    def _domain_fields():
        """The fields src/schemas/state.py declares in its OWN class body.

        Read from the source rather than from ``State.__annotations__``: a
        TypedDict merges inherited annotations, and State deliberately
        re-declares several framework fields (result, formatted_output,
        validated_input, ...) for merge compatibility — so neither the merged
        dict nor a set-subtraction against AgentState answers the question this
        guard asks, which is "what did WE add or re-declare, and is it
        classified".
        """
        source = pathlib.Path(__file__).resolve().parents[2] / "src" / "schemas" / "state.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "State":
                return {
                    stmt.target.id
                    for stmt in node.body
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                }
        raise AssertionError("class State not found in src/schemas/state.py")

    def test_every_declared_state_field_is_classified(self):
        declared = self._domain_fields()
        classified = set(_OUTPUT_BEARING_STATE_FIELDS) | set(_INERT_STATE_FIELDS)
        unclassified = declared - classified
        assert not unclassified, (
            f"State field(s) {sorted(unclassified)} are neither output-bearing nor inert. "
            "Classify them in src/nodes/post_process_node.py — an unclassified field "
            "survives every output-gate violation."
        )

    def test_the_domain_field_set_is_not_empty(self):
        """If the subtraction ever returned nothing, the totality check above
        would pass vacuously on any inventory at all."""
        assert len(self._domain_fields()) >= 15

    def test_the_two_sets_are_disjoint(self):
        assert not set(_OUTPUT_BEARING_STATE_FIELDS) & set(_INERT_STATE_FIELDS)

    def test_no_classified_field_has_been_dropped_from_state(self):
        """The other direction: a stale entry means the inventory is describing
        a field that no longer exists, and the guard above stops being total."""
        declared = self._domain_fields()
        stale = (set(_OUTPUT_BEARING_STATE_FIELDS) | set(_INERT_STATE_FIELDS)) - declared
        assert not stale, f"inventory names field(s) State no longer declares: {sorted(stale)}"


class TestDetectorParityWithTheFramework:
    """A boundary narrower than the framework's own scan is a BYPASS, not a
    narrower gate: the framework refuses the node one step later and discards
    the withholding along with it, leaving the un-gated document in state.
    """

    @pytest.mark.parametrize(
        "shape",
        [
            "sk_live_" + "a" * 20,
            "sk-" + "b" * 24,
            "eyJ" + "c" * 20,
            "AKIA" + "D" * 16,
            "Bearer " + "e" * 24,
            "postgresql://" + "user:pw@host:5432/db",
        ],
    )
    def test_blocks_every_shape_the_framework_would_refuse(self, shape):
        # Keeps the parametrization honest: each shape really is one the
        # framework refuses, so this cannot silently become a no-op list.
        assert detect_credentials(shape), "probe shape is not a credential the framework refuses"
        violation = _security_gate_output(f"HR POLICY & BENEFITS\n{shape}")
        assert violation is not None, f"framework refuses {shape!r} but the boundary passed it"
        assert shape not in violation

    def test_keeps_the_assignment_shape_the_framework_lacks(self):
        """The union, not a swap: delegating must not DROP a check either."""
        assert detect_credentials("password = supersecret123") == []
        assert _security_gate_output("password = supersecret123") == "credential_assignment"

    def test_ordinary_policy_text_is_not_flagged(self):
        assert _security_gate_output(_DOCUMENT) is None

    def test_the_agent_class_mirror_is_the_same_scan(self):
        """The agent-class gate and the runtime boundary must not drift — they
        were two copies once, and both were narrower than the framework."""
        agent = HRPolicyBenefitsQAAgent()
        for probe in ("AKIA" + "D" * 16, "sk_live_" + "a" * 20, "password = supersecret123", _DOCUMENT):
            assert agent._security_gate_output(probe) == _security_gate_output(probe)


# ---------------------------------------------------------------------------
# Half 2 — the envelope withholds on status
# ---------------------------------------------------------------------------


def _envelope(**overrides):
    state = {
        "status": AgentStatus.SUCCESS.value,
        "formatted_output": _DOCUMENT,
        "result": _DOCUMENT,
        "answer": "Full-time employees accrue 20 days of annual paid leave.",
        "answer_document": _DOCUMENT,
        "citations": to_json([{"doc_id": "leave-annual-001"}]),
        "reranked_docs": to_json([{"doc_id": "leave-annual-001", "text": "Full-time employees accrue 20 days"}]),
        "entitlement_summary": to_json({"records": [{"category": "Annual Paid Leave"}]}),
        "node_history": [],
    }
    state.update(overrides)
    return HRPolicyBenefitsQAAgent().get_output(state)


class TestEnvelopeSuccessControl:
    """Without this, every containment assertion below would also pass on an
    envelope that returns nothing at all."""

    def test_success_surfaces_the_boundary_output_and_the_domain_fields(self):
        envelope = _envelope()
        assert envelope["status"] == AgentStatus.SUCCESS.value
        assert envelope["output"] == _DOCUMENT
        assert envelope["result"] == _DOCUMENT
        assert envelope["formatted_output"] == _DOCUMENT
        for key in ("answer", "citations", "answer_document", "reranked_docs", "entitlement_summary"):
            assert envelope[key] is not None, f"{key} must be surfaced on the success path"
        for key in ("trace_id", "correlation_id", "node_history"):
            assert key in envelope, "the base envelope keys must survive the extension"


class TestEnvelopeContainment:
    def test_pre_boundary_result_is_never_surfaced_on_an_error(self):
        """formatted_output is the POST-boundary value and is surfaced on both
        paths by design; `result` is the PRE-boundary one and must not be."""
        notice = "[HR ANSWER WITHHELD: the assembled answer did not satisfy the output contract.]"
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=notice)
        assert envelope["result"] is None
        blob = json.dumps(envelope, ensure_ascii=False)
        for needle in _RELEASED:
            assert needle not in blob

    def test_the_or_result_fallback_is_dead_on_an_error(self):
        """The hole in the base envelope: with no formatted_output, `output`
        falls through to the pre-boundary `result`. This is the path taken when
        PostProcessNode raises — its withholding return is discarded, so state
        still holds the document and formatted_output was never written."""
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=None)
        assert not envelope["output"]
        blob = json.dumps(envelope, ensure_ascii=False)
        for needle in _RELEASED:
            assert needle not in blob

    def test_error_surfaces_the_boundary_notice_and_nothing_else(self):
        notice = "[HR ANSWER WITHHELD: ...]"
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=notice)
        assert envelope["output"] == notice
        assert envelope["formatted_output"] == notice
        assert envelope["result"] is None

    def test_structured_fields_are_withheld_on_an_error(self):
        envelope = _envelope(status=AgentStatus.ERROR.value)
        for key in ("answer", "citations", "answer_document", "reranked_docs", "entitlement_summary"):
            assert envelope[key] is None, f"{key} released on the error path"

    @pytest.mark.parametrize(
        "status",
        [AgentStatus.TIMEOUT.value, AgentStatus.CANCELLED.value, AgentStatus.RETRY.value],
    )
    def test_containment_holds_for_every_non_success_status(self, status):
        """Not an ERROR special case. These are exactly the statuses the
        backbone routes from `main` STRAIGHT to finalize — PostProcessNode
        never runs, so the envelope is the only guard left standing."""
        envelope = _envelope(status=status, formatted_output=None)
        assert envelope["result"] is None
        assert not envelope["output"]
        blob = json.dumps(envelope, ensure_ascii=False)
        for needle in _RELEASED:
            assert needle not in blob

    def test_no_output_bearing_field_reaches_the_error_envelope(self):
        """Ties the inventory to the caller-visible surface: nothing classified
        output-bearing may appear in a non-success envelope under any key."""
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=None)
        for field in _OUTPUT_BEARING_STATE_FIELDS:
            assert not envelope.get(field), f"{field} surfaced on the error envelope"
