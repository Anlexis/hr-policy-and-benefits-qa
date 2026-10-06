"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, the platform gateway calls agent.invoke() directly.

import json
import os
import secrets
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import Graph, _runtime_config

app = FastAPI(title="HRPolicyBenefitsQAAgent")

# The platform registry loads config/config.yaml and passes it as
# Graph(config=...); the standalone server mirrors that exactly so the declared
# runtime parameters (max_retry, the retrieval settings, the entitlement
# grading rules) are live in both deployments.
agent = Graph(config=_runtime_config())
agent.compile()
# Namespace and agent name match the manifest values in config/agent.yaml.
agent.provision_secrets(secrets_factory(namespace="svc", agent_name="HRPolicyBenefitsQAAgent"))

# Upper bound on the serialized caller-data channel, in bytes. The pre_process
# node enforces the per-field bounds (channel alphabet, entry cap, day-count
# range); this is the coarse adapter-level guard that keeps an oversized
# payload from reaching the graph at all.
_MAX_INPUT_CONTEXT_BYTES = 262_144


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured invocation parameters: the channel identifier and the
    # employee's entitlement records — each validated field by field inside the
    # graph (see src/nodes/pre_process_node.py).
    input_context: Optional[Dict[str, Any]] = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone-deployment caller auth: when INVOKE_AUTH_TOKEN is set on the
    # server environment, callers that no upstream middleware vouched for
    # (still ANONYMOUS) must present it as a bearer token and then run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary — a deployment-level caller
    # credential, not an agent secret, so the secrets provider does not apply
    # (no InvocationContext exists before auth).
    #
    # Required here specifically: the pre_process node declares
    # required_trust_level = VERIFIED_EXTERNAL. Nothing else sets
    # request.state.trust_level in a standalone deployment, so without this
    # boundary every invoke arrives ANONYMOUS, the trust gate denies it, and the
    # agent returns an error.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of a clean 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — never reveal whether the token was
            # absent, malformed or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    input_context = req.input_context or {}
    if input_context and len(json.dumps(input_context, default=str)) > _MAX_INPUT_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail="input_context exceeds the maximum allowed size.")
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return agent.invoke(req.input, ctx=ctx, input_context=input_context)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "HRPolicyBenefitsQAAgent"}
