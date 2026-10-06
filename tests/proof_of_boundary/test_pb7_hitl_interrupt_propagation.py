"""PB-7: human-in-the-loop interrupt propagation.

Skip stub. This template is a single-turn retrieval pipeline and wires no
cross-boundary interrupt propagation, so the boundary has nothing to assert yet.
The stub keeps the boundary suite complete and turns into a real test the moment
that wiring is added: the guard below introspects for it rather than asserting a
constant.
"""

import pytest


def _propagate_hitl_wired() -> bool:
    """Return True if the graph wires cross-boundary HITL interrupt propagation."""
    # This template is a stateless single-turn pipeline, so no propagation is
    # implemented. When that wiring is added, replace this guard with the real
    # introspection (e.g. inspect the compiled graph for an interrupt path).
    return False


@pytest.mark.skipif(
    not _propagate_hitl_wired(),
    reason=(
        "this template is a stateless single-turn pipeline and wires no "
        "cross-boundary interrupt propagation; the test is skipped until that "
        "wiring is added to the domain workflow graph."
    ),
)
def test_pb7_hitl_interrupt_propagation():
    """PB-7: Verify that HITL interrupt signals propagate correctly across the
    backbone boundary (initialize → pre_process → [HITL interrupt] → resume).

    When this test runs (once HITL wiring is live), it must:
    - Compile the graph
    - Invoke with a payload that triggers the HITL interrupt path
    - Assert that the interrupt signal reaches the caller (hitl_* fields in state)
    - Assert that resuming from the interrupt produces the correct final output

    Until HITL propagation is wired, this test is skipped by the guard above.
    """
    raise AssertionError(
        "PB-7: implement the real interrupt-propagation assertions when "
        "cross-boundary human-in-the-loop wiring is added to this template."
    )
