# HR Policy & Benefits Q&A Agent

AI agent for answering HR policy and benefits questions, built with Agentic Star.

> **Category**: Cat 2 (a domain pipeline: a retrieval workflow behind the fixed agent backbone)
> **Industry**: Services
> **Template ID**: SVC-C2-001

## Overview

Answers an employee's question about leave, benefits, expenses and work-style rules from an HR
policy corpus, quoting only the passages that clear a relevance threshold and citing each one. When
no passage clears it the agent says so rather than guessing.

A caller may also submit the employee's own leave-entitlement records alongside the question. The
answer then states the position those records put them in — days entitled, taken and remaining per
category — and flags the categories where the statutory uptake has not been met or the balance
exceeds the carry-over cap.

Retrieval and answer composition are deterministic and offline: the shipped corpus is scored by
content-term recall and the answer is assembled from the retrieved text, so the same question always
returns the same answer. Swapping in a vector store and a constrained model is a configuration
change, not a rewrite.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/02_design.md` for the architecture, the caller contract and the output boundary, and
`docs/03_test_spec.md` for what the shipped test suite covers.

## Customising

1. Adjust `config/config.yaml` for your own environment: the retrieval settings and the
   entitlement grading rules (statutory uptake trigger and minimum, carry-over cap).
2. Replace the policy corpus in `src/nodes/retrieve_node.py` with your own policy set, or point the
   node at your vector store.
3. Review the caller contract in `src/nodes/pre_process_node.py` — the fields, bounds and screens
   your callers will be held to — and the output boundary in `src/nodes/post_process_node.py`.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

