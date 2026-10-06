# Design Specification — SVC-C2-001 HR Policy & Benefits Q&A

## Position in the architecture

| Aspect | Value |
|---|---|
| Agent class | `HRPolicyBenefitsQAAgent` (`src/graph/graph.py`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Category / pattern | Cat 2 — a domain pipeline: a retrieval workflow behind the fixed agent backbone |
| Architecture | Two-layer nested graph |
| Outer backbone | `AgentBaseGraph` (fixed five-node skeleton — `add_edges()` is never overridden) |
| Inner workflow | `DomainWorkflowGraph` (`BaseGraph`) wired into the `main` slot through a `GraphNode` |

Three-layer separation:

- **State** — a flat `TypedDict`; dict/list fields are stored JSON-serialised because the
  checkpointer serialises state with msgpack.
- **Node** — framework inheritance; a domain node overrides `execute(self, state) -> dict` and
  nothing else.
- **Graph** — composition; `register_nodes()` fills the slots.

## Architecture overview

### Outer backbone

| Node | Responsibility | Input state | Output state | Class |
|------|---------------|-------------|--------------|-------|
| initialize | Framework defaults (schema version, session id, trust level) | user_input | — | framework default |
| pre_process | **Trust gate** (VERIFIED_EXTERNAL) + the caller-data contract | user_input, input_context | validated_input, entitlement_records, channel | `PreProcessNode` |
| main | Delegates to the inner retrieval workflow | validated_input, entitlement_records | answer, citations, answer_document, entitlement_summary | `HRPolicyQAGraphNode` (a `GraphNode`) |
| post_process | **The output boundary** (four layers, below) | result / answer_document, reranked_docs, citations, entitlement_summary | formatted_output, result | `PostProcessNode` |
| finalize | Framework defaults (response metadata, total time) | — | — | framework default |

### Inner workflow (`DomainWorkflowGraph`)

| Node | Responsibility | Input → Output |
|------|---------------|----------------|
| input_validate | Normalise the question; publish the validated records into the inner state | validated_input, input_context → normalized_query, entitlement_records |
| retrieve | Score the policy corpus by content-term recall and keep the top-k | normalized_query → retrieved_docs |
| rerank_filter | Sort by relevance and drop everything below `score_threshold` | retrieved_docs → reranked_docs |
| generate_answer | Compose the answer strictly from the kept passages, with citations; abstain when none clear the threshold. Compute the entitlement position from the validated records | reranked_docs, entitlement_records → answer, citations, entitlement_summary |
| output_format | Assemble the answer document | answer, citations, entitlement_summary → answer_document, result |

### Data flow

```
Outer:  START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
                                              |
                                              +-- (retry, up to max_retry) -> pre_process

Inner (main slot -> DomainWorkflowGraph):
   START -> input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format -> END
```

## The caller contract

A caller submits a question and, optionally, the employee's own entitlement records. Both channels
are validated in `PreProcessNode`, which is the only node declaring VERIFIED_EXTERNAL trust.

| Field | Channel | Contract |
|---|---|---|
| the question | `input` | non-empty, ≤ 2000 characters after control characters are stripped, refused if it carries instruction-override content. A JSON envelope carrying it under `question` / `query` / `text` is accepted |
| `channel` | `input_context` | an inert identifier, `[a-z0-9_]{1,32}`; defaults to `unknown` |
| `entitlement_records` | `input_context` | a list, entry-capped by `qa.max_entitlement_records` |
| `…[].category` | `input_context` | label alphabet, ≤ 60 characters |
| `…[].policy_reference` | `input_context` | label alphabet, ≤ 80 characters; optional |
| `…[].days_entitled` | `input_context` | a finite number in [0, 400] |
| `…[].days_taken` | `input_context` | a finite number in [0, 400], not above `days_entitled` |

Rules that hold across every field:

- **Fail closed.** A field that violates its contract refuses the whole request; nothing is
  repaired, truncated or defaulted into range.
- **Never echo a rejected value.** An error names the field and restates the contract.
- **Every number is finite.** `NaN` and `±Infinity` parse through `float()` and arrive intact in a
  raw JSON body, and every comparison against `NaN` is False — a `NaN` day count would slide under
  the statutory-uptake threshold and be reported as on track. They are refused like any other
  out-of-contract value.
- **Caller strings that render are inert.** The label alphabet is word characters in any script
  (so Japanese category names are accepted), space, the section sign and the punctuation policy
  references use. Quotes, angle brackets, colons, at-signs, braces, control characters and every
  newline are refused, so a label can neither carry markup nor break the document's line structure.
- **Contact identifiers are refused, not masked.** Email addresses, phone numbers and
  national/payment identifiers are screened out of every label. Personal-name heuristics are
  deliberately not screened on: they match any two consecutive capitalised words, which is what a
  leave category is.

**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that an answer was produced. A value the caller can correct (an
out-of-contract parameter, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set, the structured output
fields are withheld, and `PostProcessNode` renders the reason as a static
caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

### Why the records travel on the caller-data channel

`GraphNode.extract_input()` returns a string the framework writes into the inner state under
`user_input` — a field the framework's input gate rewrites, whose personal-name heuristic matches
any two consecutive capitalised words. That is exactly the shape of a leave category ("Annual Paid
Leave", "Nursing Care Leave"), so a record carried there would arrive with its labels replaced by a
mask token and the answer would name the wrong entitlements.

Only the question travels on `user_input`. The validated records cross on the caller-data channel,
which carries structured values through untouched — and because that channel is not scanned by the
platform's masking gate, the template screens those fields itself (above). Moving data off the
scanned field without adding that screen would be a fail-open, not a fix.

The framework does not forward the caller-data channel into a subgraph, so `src/graph/context_bridge.py`
bridges it: the outer `extract_input()` stashes the validated record immediately before the inner
invoke, and the inner graph's `_extra_initial_state()` restores it. Only already-validated values
are stashed — the bridge is a transport, never a second contract.

## The output boundary

The document the caller receives states its own disclosure schema, and `PostProcessNode` enforces
it in four independent layers:

1. **Credential scan** — an API key, JWT, bearer token, cloud access key, connection string or
   credential assignment anywhere in the document withholds the response entirely (error status,
   stub body). Recognition is delegated to the framework's own `detect_credentials()`, unioned with
   the credential-ASSIGNMENT shape the framework does not carry — never a local pattern set. The
   framework scans every node result with that same detector and *raises* on a hit, and a raise
   discards the node's whole return **including the containment below**; so a shape the framework
   catches and this boundary missed would be a containment bypass, not merely a narrower gate. The
   two cannot drift apart by construction. `HRPolicyBenefitsQAAgent._security_gate_output()`
   delegates to the same module-level function rather than keeping a second copy.
2. **Grounding** — policy text is quoted only from passages that cleared the relevance threshold,
   and every one of them is cited. A citation no kept passage backs, or a citation rendered when no
   passage cleared the threshold, withholds the response. The check reads the passage list
   independently of the node that produced the citations.
3. **Caller-label containment** — every rendered entitlement label is re-validated against the same
   alphabet the input contract enforces, and a label that fails is withheld wherever it appears.
4. **Contact-identifier redaction** — an email address, phone number or national/payment identifier
   reaching the document is replaced verbatim with a fixed token, with an audit event recording how
   many were replaced.

**Layer order.** The credential scan is a pattern scan, so it runs before any rewrite — a rewrite
landing inside a credential-shaped string would destroy the very pattern the scan looks for. Label
containment is a verbatim replacement and is order-independent; identifier redaction is a pattern
scan and runs after it. Both pattern scans are then repeated on the rewritten document, so nothing
a rewrite produced can escape a layer that ran before it.

**Withholding CONTAINS, it does not merely refuse.** `AgentBaseGraph.get_output()` resolves the
caller-facing `output` key as `formatted_output or result` *without consulting status*, and `result`
is the value `merge_output()` mapped out of the inner workflow — pre-boundary whenever
`PostProcessNode` did not itself write it. A boundary that returned ERROR while leaving the
assembled answer, its citations, the retrieved passages and the entitlement position in state would
therefore ship the refused content inside the error envelope. Two halves close this:

- **the node** overwrites every field named in `_OUTPUT_BEARING_STATE_FIELDS` (`answer`,
  `answer_document`, `citations`, `reranked_docs`, `retrieved_docs`, `entitlement_summary`,
  `entitlement_records`, `validated_input`, `normalized_query`, `result`, `formatted_output`), and
  the `formatted_output` replacement is deliberately **non-empty** so the `or result` fallback
  cannot re-open the path it just closed. Inert fields (counts, channel, identifiers, validated
  configuration) may survive, but the two sets must together cover every field `src/schemas/state.py`
  declares — `tests/unit/test_output_containment.py` asserts that partition is total, so a field
  added later cannot quietly join the survivors.
- **the envelope** (`HRPolicyBenefitsQAAgent.get_output()`) surfaces `result` only on the gated
  success path and re-resolves `output` as `formatted_output or None` on any non-success outcome.
  This half is not redundant: `PostProcessNode` is skipped entirely when the backbone routes a
  non-success `main` straight to finalize, and its return is discarded wholesale when it raises or
  when the framework's output-side credential scan refuses its result — on each of those paths the node's containment
  never runs at all, and status is the only thing left to gate on.

Violation messages name **locations and finding types only**, never the matched value: echoing it
would put the refused string back into the node's own result, where the framework's output-side scan
raises and discards the containment with it.

**No monetary rounding grid applies.** This template computes and renders no monetary value. Day
counts are exact figures taken from the caller's own records, with the remaining balance derived
from them, so they are rendered as submitted rather than snapped to a disclosure grid.

## Runtime configuration

`config/agent.yaml` is the flat registration manifest: identity, entry point, required trust level
and the compile-time `requires` block. It holds no runtime values.

`config/config.yaml` holds every runtime parameter. The platform registry loads it and passes it as
`Graph(config=...)`; `src/api/server.py` reads the same file and passes it the same way, so a
standalone deployment and a registry-loaded agent behave identically.

| Key | Default | Consumer |
|---|---|---|
| `max_retry` | 3 | the framework's retry routing |
| `timeout_s` | 30 | the deployment's request budget |
| `retrieval.collection` | `svc_hr_policy_and_benefi_kb` | `RetrieveNode` |
| `retrieval.top_k` | 5 | `RetrieveNode` |
| `retrieval.score_threshold` | 0.70 | `RerankFilterNode` |
| `retrieval.hybrid_search` | false | reserved for a vector-store retriever |
| `qa.max_entitlement_records` | 20 | the entry cap in `PreProcessNode` |
| `qa.statutory_trigger_days` | 10 | `GenerateAnswerNode` grading |
| `qa.statutory_minimum_days` | 5 | `GenerateAnswerNode` grading |
| `qa.carryover_cap_days` | 40 | `GenerateAnswerNode` grading |
| `llm.*` | — | not forwarded: answer generation is deterministic and invokes no model |

`declared_settings()` validates every value for type, finiteness and range when the graph is built.
An invalid or absent key is not forwarded and the consuming node keeps its module default, so a
malformed configuration file can neither crash graph construction nor weaken a guard. The node
contract is `execute(self, state) -> dict` and takes no config argument, so the validated settings
reach the nodes through state — seeded into the outer initial state for the pre_process bounds, and
forwarded through the main slot into the inner graph for the pipeline.

## State definition (domain fields — `src/schemas/state.py`)

| Field | Type | Purpose |
|-------|------|---------|
| validated_input | NotRequired[Optional[str]] | normalised question from pre_process |
| enriched_context | NotRequired[Optional[str]] | JSON channel metadata |
| entitlement_records | NotRequired[Optional[str]] | JSON list of validated caller records |
| record_count | NotRequired[Optional[int]] | number of validated records |
| channel | NotRequired[Optional[str]] | the caller's inert channel identifier |
| runtime_settings | NotRequired[Optional[str]] | JSON validated settings seeded into the outer state |
| retrieval_config | NotRequired[Optional[str]] | JSON retrieval settings seeded into the inner state |
| qa_config | NotRequired[Optional[str]] | JSON grading rules seeded into the inner state |
| normalized_query | NotRequired[Optional[str]] | cleaned query for retrieval |
| retrieved_docs | NotRequired[Optional[str]] | JSON list of retrieved passages |
| reranked_docs | NotRequired[Optional[str]] | JSON list of passages above the threshold |
| answer | NotRequired[Optional[str]] | grounded answer text |
| citations | NotRequired[Optional[str]] | JSON list of `{doc_id, source}` |
| entitlement_summary | NotRequired[Optional[str]] | JSON position computed from the records |
| answer_document | NotRequired[Optional[str]] | assembled answer document |
| result / formatted_output | NotRequired[Optional[str]] | caller-facing values, after the output boundary |

State constraints:

- flat `TypedDict` only (primitives and JSON-serialisable values); dict/list fields stored as JSON
  `Optional[str]`;
- no credentials, tokens or keys in state — it is checkpointed;
- no Pydantic models, dataclasses or arbitrary Python objects (msgpack-incompatible).

## Framework usage

- `emit_trace_event()` — at least one domain audit event inside every `execute()`.
- Trust gate — `required_trust_level = VERIFIED_EXTERNAL` on the pre_process slot; every other node
  is ANONYMOUS, so the passthrough context is accepted at the subgraph boundary.
- Output gate — `_security_gate_output` is declared on the agent class, and `PostProcessNode`
  enforces the identical contract at runtime through a module-level function called from inside
  `execute()`. It is never an instance method: the framework's own gate methods are final, and a
  node-level override raises at class definition.

Gate behaviour by node type:

- `FunctionNode` subclass → the framework's input and output gates always run automatically;
- `GraphNode` (the main slot) → a deliberate no-op, since the inner nodes' gates already ran.

## Import isolation

- The template imports `framework/`, `shared/`, `langgraph` and its own `src/` — nothing else.
- It never imports the platform SDK the framework itself sits on.

## Design decisions

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base class | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | a question-answering pipeline is a fixed backbone, not an autonomous loop |
| Composition | a single node | a nested subgraph | **nested subgraph** | retrieval needs a multi-step retrieve → rerank → generate → format topology |
| Retrieval | an external vector store | a shipped in-module corpus | **shipped corpus** | keeps the template runnable and deterministic offline; the grounding and threshold contract is identical when a vector store is wired in |
| Answer generation | a constrained model | deterministic grounded synthesis | **deterministic synthesis** | the answer is composed strictly from the retrieved passages with citations, which is the same grounding and abstention contract a model deployment must hold |
| Caller records channel | on `user_input` | on the caller-data channel + a template-owned screen | **caller-data channel** | the input gate rewrites `user_input` and would mangle the labels; the channel it moves to is screened by the template itself |

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.
