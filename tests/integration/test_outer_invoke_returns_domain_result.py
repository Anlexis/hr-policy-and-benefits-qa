# Integration: the compiled OUTER graph's success path must return the answer to
# the caller as data, not only as rendered text.
#
# The framework's own get_output() surfaces just the {output, status, ...}
# envelope, so an agent that does not extend it drops its structured result —
# answer, citations, answer_document, formatted_output — even though the nodes
# populated them. A test that reads only ["output"] never notices.
#
# Case 1 invokes the compiled outer graph as a real VERIFIED_EXTERNAL caller and
# asserts the answer is surfaced. Case 2 proves the override is fail-closed: a
# credential echoed into the assembled document withholds the response, and the
# structured fields stay withheld while the caller only ever sees the stub.

from framework.schemas.agent_status import AgentStatus
from src.schemas.state import from_json

_GROUNDED_QUERY = (
    "How many days of annual paid leave do full-time employees accrue each "
    "fiscal year, and how many days must be taken?"
)


def _patch_domain_emit(monkeypatch):
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
            monkeypatch.setattr(f"src.nodes.{mod_suffix}.emit_trace_event", lambda *a, **k: None)
        except AttributeError:
            pass


def _invoke(monkeypatch, question):
    _patch_domain_emit(monkeypatch)
    from framework.schemas.invocation_context import InvocationContext, TrustLevel
    from src.graph.graph import Graph, _runtime_config

    agent = Graph(config=_runtime_config())
    agent.compile()
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    return agent.invoke(question, ctx=ctx)


class TestOuterInvokeReturnsDomainResult:
    def test_success_invoke_surfaces_domain_answer(self, monkeypatch):
        result = _invoke(monkeypatch, _GROUNDED_QUERY)

        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"expected SUCCESS, got {result.get('status')!r}; error_log={result.get('error_log')}"
        # Without the get_output() override these are all None.
        assert result.get("formatted_output") is not None, "formatted_output must be surfaced"
        assert result.get("result") is not None, "result must be surfaced"
        assert result.get("answer") is not None, "answer must be surfaced"
        assert result.get("answer_document") is not None, "answer_document must be surfaced"
        # Grounded + cited.
        assert "HR POLICY & BENEFITS" in result["formatted_output"]
        assert "HR Policy Manual" in result["answer_document"]
        assert from_json(result.get("citations")), "citations must be surfaced on the grounded answer"
        # Framework envelope preserved (backward compatible).
        assert result.get("output") is not None

    def test_credential_block_withholds_structured_fields(self, monkeypatch):
        """Fail-closed: a credential echoed into the document withholds the response
        and the structured fields with it."""
        # The question is echoed into the assembled document, so a credential in
        # it must trip the output boundary in PostProcessNode.
        leaky = "What is the annual leave policy? api_key=sk-abcdefghij0123456789ABCDEF"
        result = _invoke(monkeypatch, leaky)

        assert (
            result.get("status") == AgentStatus.ERROR.value
        ), f"expected the output boundary to withhold, got status={result.get('status')!r}"
        # Raw structured domain fields are withheld on the non-success path.
        assert result.get("answer") is None
        assert result.get("citations") is None
        assert result.get("answer_document") is None
        assert result.get("reranked_docs") is None
        # The caller-facing output carries only the stub — never the secret.
        assert "sk-abcdefghij0123456789ABCDEF" not in (result.get("formatted_output") or "")
        assert "sk-abcdefghij0123456789ABCDEF" not in (result.get("output") or "")
