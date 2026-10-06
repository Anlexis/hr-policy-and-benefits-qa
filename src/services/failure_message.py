"""AgentCore Platform v1.0 — caller-facing wording for a rejected run.

A failed run reaches the caller as a terminal failure whose text the platform runner produces: it
names the exception type and nothing else. The rejecting node therefore sends the reason itself, as
a non-terminal event, before returning the error — and it has to happen there, because once a run
carries ``status: error`` the framework skips ``execute()`` on every later node.

The wording is written for the person who sent the request, so it does not reuse the node's own
reason text: that text is written for an operator reading a log and names internal state keys and
design terms. Each sentence for a caller-fixable rejection says what to change; the sentences for
conditions the caller cannot change say so instead of inviting a pointless retry.

Sentences are static — no request value, record value, or internal identifier is ever substituted,
because a progress event leaves the process and is not covered by the S-3 output gate.
"""

# ── Caller-fixable: the request itself needs changing ────────────────────────
EMPTY_INPUT = "No question was received. Send the question you want answered."
TOO_LONG = "The request is too long. Shorten it and send it again."
TOO_SHORT = "The question is too short. Add more detail and send it again."
NOT_JSON = "The request is not valid JSON. Check for a missing quote, comma, or brace."
NOT_OBJECT = "The request must be a JSON object, not a list or a plain value."
MISSING_FIELDS = "The request is missing a required field. Check it against the documented format."
INVALID_VALUE = "A value in the request could not be accepted. Check it against the documented format."
DISALLOWED_FORM = "The request contains an instruction form that is not accepted. Send the question as plain text."
NOTHING_TO_ACT_ON = "The request does not say what to act on. Name the target and send it again."
INPUT_REJECTED = "The request could not be accepted. Check it against the documented format and send it again."

# ── Not caller-fixable: retrying the same request will not help ──────────────
RETRIEVAL_FAILED = "The search could not be completed. Try rephrasing the question."
PROCESSING_FAILED = "The request could not be processed. This is not something the request can fix."
OUTPUT_BLOCKED = "The response could not be delivered. This is not something the request can fix."
