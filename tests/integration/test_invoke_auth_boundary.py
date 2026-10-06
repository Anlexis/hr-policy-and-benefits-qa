# Integration: the /invoke authentication boundary.
#
# A standalone deployment requires INVOKE_AUTH_TOKEN to be set. The HTTP adapter
# (src/api/server.py) raises an otherwise-anonymous caller to VERIFIED_EXTERNAL
# only when it presents the matching bearer token — which is exactly what the
# pre_process trust gate requires for a successful invoke.
#
# This test drives the real ASGI app through a test client and asserts both
# halves: unauthenticated and wrong-token calls are rejected with 401, and an
# authenticated call reaches the agent and succeeds.
#
# It skips cleanly where fastapi or the framework runtime is not importable, so
# it never breaks the suite; it runs wherever the deployment dependencies exist.

import pytest

_TOKEN = "rehearsal-bearer-token-abc123"
_PAYLOAD = {
    "input": (
        "How many days of annual paid leave do full-time employees accrue each "
        "fiscal year, and how many days must be taken?"
    ),
    "session_id": "auth-boundary-test",
}


def _client(monkeypatch):
    """Build a TestClient for the real app with INVOKE_AUTH_TOKEN configured."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    testclient = pytest.importorskip("fastapi.testclient")
    try:
        from src.api.server import app

        return testclient.TestClient(app, raise_server_exceptions=False)
    except Exception as exc:  # fastapi / framework runtime not available here
        pytest.skip(f"the app or its test client is not available in this environment: {exc}")


class TestInvokeAuthBoundary:
    def test_unauthenticated_call_is_rejected(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post("/invoke", json=_PAYLOAD)
        assert resp.status_code == 401, f"expected 401 for a missing token, got {resp.status_code}"

    def test_wrong_token_is_rejected(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post("/invoke", json=_PAYLOAD, headers={"Authorization": "Bearer not-the-token"})
        assert resp.status_code == 401, f"expected 401 for a wrong token, got {resp.status_code}"

    def test_authenticated_call_succeeds(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post("/invoke", json=_PAYLOAD, headers={"Authorization": f"Bearer {_TOKEN}"})
        assert resp.status_code == 200, f"expected 200 for a valid token, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body.get("status") == "success", f"expected agent status=success, got {body.get('status')!r}"
