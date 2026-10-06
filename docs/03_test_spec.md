# Test Specification — SVC-C2-001 HR Policy & Benefits Q&A

## 1. Test strategy

- **Agent:** SVC-C2-001 — HR Policy & Benefits Q&A. A Cat 2 domain pipeline: a two-layer nested
  graph, outer `AgentBaseGraph` backbone plus inner `DomainWorkflowGraph`.
- **Coverage target:** ≥ 90% of the branches in `src/nodes/` and `src/graph/`.
- **Test types:** unit (per node, per graph, plus the caller contract and the output boundary) ·
  proof-of-boundary (framework security and serialisation contracts, and the HTTP entry point) ·
  integration (the compiled outer graph and the authentication boundary).
- **Framework provisioning:** `agenticstar-agentcore` is supplied by the build. The tests import
  the real modules; there are no stub nodes.
- **Audit events:** `emit_trace_event` is patched at the node module level, never via a
  `sys.modules` stub — that would break the real `shared` package the framework loads at import
  time.

### The grounding and abstention contract

The business value of this template is a **grounded** answer: policy text is quoted strictly from
passages that cleared the relevance threshold (0.70), and each is cited. When nothing clears the
threshold the agent **abstains** with an explicit statement that no policy covered the question —
it never fabricates policy. The suite asserts both halves, and the output boundary enforces the
contract independently of the node that produced the citations.

### The caller-data contract

A caller may submit the employee's entitlement records alongside the question. Every field is
validated against explicit bounds and fails closed; the answer then reports a position computed
from those records. The suite probes that contract in both directions — hostile input refused,
legitimate HR text accepted.

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | every node plus outer and inner graph wiring; grounding and abstention |
| `tests/unit/test_caller_data_and_output_gate.py` | the caller contract (accept and refuse), the entitlement arithmetic, and the four output-boundary layers |
| `tests/unit/test_config_propagation.py` | the config file → declared settings → inner graph → node path, end to end |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | the framework's node gates are non-bypassable |
| `tests/proof_of_boundary/test_invoke_e2e.py` | PB-8: the real HTTP entry point — auth, caller data reaching the inner graph, every grade path, rejection, and the returned document's schema |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6: per-node and backbone invoke order, the trust gate, and payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4: import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2 / PB-5: state serialisation and checkpoint safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7: skip stub — no cross-boundary interrupt propagation is wired |
| `tests/integration/test_outer_invoke_returns_domain_result.py` | the compiled outer graph surfaces the answer as data, and withholds it when the boundary blocks |
| `tests/integration/test_invoke_auth_boundary.py` | bearer authentication on the HTTP entry point |

### The canonical request

The question used by the backbone invoke test and by `deploy/invoke_payload.json` — the two must
stay identical, which `test_invoke_payload_matches_pb6` asserts:

```text
How many days of annual paid leave do full-time employees accrue each fiscal
year, and how many days must be taken?
```

Every salient content term of this question overlaps the annual paid-leave passage (`HR Policy
Manual §4.1 Annual Paid Leave`), so its relevance score is 1.0 ≥ 0.70: the passage is kept and
cited. The payload also carries two entitlement records, and
`test_invoke_payload_caller_data_satisfies_the_input_contract` runs them through the input contract
so a deployment's first invoke can never fail on its own shipped payload.

## 2. Framework compliance tests

| TC-ID | Test | Expected result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, domain fields `NotRequired`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Invalid or empty input rejected at the pre_process slot | error status, error_log populated | `TestInputContractRefuses` |
| TC-03 | No credential material in State | credential scan: 0 violations | `test_state_safety.py` |
| TC-04 | `execute(self, state)` contract | signature is `(self, state)` | `test_execute_signature_is_state_first` |
| TC-05 | An audit event inside every node `execute()` | ≥ 1 domain event per node | asserted per node in `test_nodes.py` |
| TC-06/07 | The framework's node gates cannot be overridden | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | Trust gate enforced before `execute()` | ANONYMOUS caller refused; VERIFIED_EXTERNAL admitted | `TestTrustGate` |
| TC-08a | pre_process is VERIFIED_EXTERNAL; every other node ANONYMOUS | trust level asserted per node | `test_trust_level_*` |
| TC-11 | The output boundary on post_process | credential → withheld with error status; clean → passes | `TestOutputBoundary` |
| TC-12 | Withholding CONTAINS the document | every field in `_OUTPUT_BEARING_STATE_FIELDS` overwritten; the `formatted_output` replacement is TRUTHY so the envelope's `formatted_output or result` fallback cannot resolve to the pre-boundary value; no released string, traceback or source path survives; the clearing return survives the framework's own output-side credential scan | `TestWithholdContainsTheDocument` |
| TC-13 | Output-bearing inventory is TOTAL | every field `src/schemas/state.py` declares (read from source by AST) is classified output-bearing or inert, the sets are disjoint, and neither names a field State no longer declares — so a field added later cannot quietly survive a violation | `TestOutputBearingInventoryIsTotal` |
| TC-14 | Detector parity with the framework | every shape the framework's `detect_credentials()` refuses (`sk_live_`, `sk-`, `eyJ`, `AKIA`, `Bearer`, connection strings) is also refused by the boundary, with each probe asserted to be a real framework finding; the credential-ASSIGNMENT shape the framework lacks is still caught; ordinary policy text is not flagged; the agent-class gate returns the identical verdict | `TestDetectorParityWithTheFramework` |
| TC-15 | The invoke envelope withholds on status | success control surfaces the boundary output and every domain field; on ERROR / TIMEOUT / CANCELLED / RETRY `result` is None, `output` re-resolves to `formatted_output or None`, and no output-bearing field reaches the envelope under any key | `TestEnvelopeSuccessControl`, `TestEnvelopeContainment` |

## 3. Proof-of-boundary tests

| PB-ID | Boundary | Expected result | Where |
|-------|----------|----------------|-------|
| PB-2 | State serialisation | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | 0 platform-SDK imports under `src/` | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields or prohibited types in State | `test_state_safety.py` **Auto-waived — checkpointing disabled**: `config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`, so no checkpoint surface exists; the conditional gate and the non-lossy traversal helper ship with the stub. |
| PB-6 | Invoke execution order (per node) | trust gate → node_start → input gate → `execute()` → output gate → node_complete, for every node under `src/nodes/` | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | node_history = `[InitializeNode, PreProcessNode, HRPolicyQAGraphNode, PostProcessNode, FinalizeNode]` | `TestBackboneInvokeOrder` |
| PB-6c | A real external caller | VERIFIED_EXTERNAL context — never an internal one; success end to end | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json` carries the PB-6 request, and its caller data passes the input contract | `test_invoke_payload_matches_pb6`, `test_invoke_payload_caller_data_satisfies_the_input_contract` |
| PB-7 | Interrupt propagation | skipped by an introspection guard — no such wiring exists yet | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | The HTTP entry point | bearer auth enforced; caller data reaches the inner graph; every grade path reachable; rejection carries no answer; the returned document satisfies its schema | `test_invoke_e2e.py` |
| PB-8b | Withheld responses are contained on the REAL surface | a clean-path control proves the same request does release the answer, then two data-path faults are injected — a drifted upstream node that renders a citation with nothing grounded (the boundary withholds), and one that hands the boundary a structured `result` so it RAISES and its containment is discarded. In both the error envelope carries no answer, citation, passage, entitlement position, traceback or source path, `result` is None, and `PostProcessNode` is in `node_history` — proving the block happened AT the boundary, not upstream | `TestWithheldResponseIsContained` |

## 4. Business-logic tests

| BL-ID | Test | Input | Expected result |
|-------|------|-------|-----------------|
| BL-01 | Grounded answer, happy path | the canonical request | assembled document with the cited `HR Policy Manual` source |
| BL-02 | Retrieval relevance | an on-topic query | top passage `leave-annual-001`, score 1.0, sorted descending |
| BL-03 | Off-topic retrieval | an off-topic query | nothing retrieved |
| BL-04 | `top_k` limit | a query matching 3 passages, `top_k=1` | exactly 1 retrieved |
| BL-05 | Threshold filter | scores `[0.90, 0.50, 0.72]` | keeps `[0.90, 0.72]` |
| BL-06 | Declared threshold | `score_threshold=0.80` | keeps only `0.90` |
| BL-07 | Grounded synthesis | one passage | the answer carries that passage's text and a citation |
| BL-08 | No fabrication | one passage | no content from any other passage appears |
| BL-09 | Abstention | no passages above the threshold | the no-coverage answer, no citations, success status |
| BL-10 | Document assembly | answer + citations | header, answer, cited source, `result == answer_document` |
| BL-11 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | the coupled keys map; `merge_output` returns changed keys only |
| BL-12 | Entitlement arithmetic | validated records | per-record remaining balance and totals computed from the records |
| BL-13 | Grade paths | records at each boundary | statutory shortfall, carry-over attention, on track — all reachable through `/invoke` |
| BL-14 | Overall grade | mixed records | the most severe record grade wins |
| BL-15 | Day-count rendering | whole and half days | rendered exactly, never rounded |

### Negative and boundary cases

| Case | Where | Expected |
|------|-------|----------|
| empty or whitespace question | PreProcessNode | error, "empty" |
| question over the length bound | PreProcessNode | error, "exceeds" |
| instruction-override content in the question or a label | PreProcessNode | error, nothing carried forward |
| ordinary question containing a screened word | PreProcessNode | accepted (the fail-closed direction) |
| `channel` outside `[a-z0-9_]{1,32}` | PreProcessNode | error naming the field |
| label outside the alphabet, or over length | PreProcessNode | error naming the field; the value never echoed |
| contact identifier in a label | PreProcessNode | refused, not masked |
| real HR labels, English and Japanese | PreProcessNode | accepted |
| non-finite / out-of-range / wrongly typed day count | PreProcessNode | error naming the field (parametrized matrix per field) |
| `days_taken` above `days_entitled` | PreProcessNode | error |
| more records than the entry cap | PreProcessNode | error naming the cap |
| oversized caller-data channel | HTTP adapter | 413 |
| empty query after normalisation | InputValidateNode | error |
| empty `normalized_query` | RetrieveNode | error |
| nothing above the threshold | RerankFilterNode | empty passage list, success |
| no passages | GenerateAnswerNode | abstention answer, success |
| empty answer | OutputFormatNode | fallback text, success |
| credential material in the document | PostProcessNode | withheld, error |
| citation no kept passage backs | PostProcessNode | withheld, error |
| label failing the contract at the boundary | PostProcessNode | withheld label, success |
| contact identifier in the document | PostProcessNode | redacted, success |
| real policy text and structural figures | PostProcessNode | byte-identical |

## 5. Execution summary

- Run with `pytest tests/`.
- Expected: every node, graph, contract and boundary test passes; the PB-7 stub is skipped by
  design.
- Coverage: both the grounded and the abstention path, every documented error path, every grade
  path, and both directions of each screen.

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `HRPolicyBenefitsQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.
