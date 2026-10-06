# SVC-C2-001 — Unit Tests: Marketplace entry point (cli.py)
#
# cli.py is not imported by any other module, so nothing else in the suite would notice if its
# import path, its graph class, or its runtime-config assembly broke — the image would build and
# then fail when the Pod starts. These exercise it directly, including the `__main__` block that
# the image's CMD actually runs.
#
# Skipped where the platform events package is absent: shared.bootstrap.marketplace_app imports it
# at module level, so cli.py cannot be imported at all in that environment.

import importlib.util
import pathlib
import runpy

import pytest

pytest.importorskip("agenticstar_platform")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CLI = _ROOT / "cli.py"


def _load():
    spec = importlib.util.spec_from_file_location("cli_under_test", _CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ != "__main__" — the runner is not started
    return module


def _config_yaml() -> dict:
    import yaml

    return yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")) or {}


def test_entry_point_imports():
    module = _load()
    assert callable(module.run_agent_marketplace)
    assert callable(module.load_agent_config)
    assert isinstance(module.HRPolicyBenefitsQAAgent, type)


def test_config_overrides_hook_is_present_and_empty():
    # extend_config is the documented seam for deployment-specific overrides. It ships empty:
    # a stray value here would silently outrank config/config.yaml on the Marketplace path only.
    module = _load()
    assert module.extend_config == {}


def test_main_hands_the_runner_the_graph_class_and_the_resolved_config(monkeypatch):
    """The image's CMD runs cli.py as __main__; this asserts what that call actually passes.

    Loading the module (the tests above) never executes the `__main__` block, so a wrong graph
    class, a dropped config or a mistyped identifier there would ship unnoticed. The runner is
    replaced before execution so nothing starts a Pod lifecycle.
    """
    import shared.bootstrap.marketplace_app as marketplace_app

    from src.graph.graph import HRPolicyBenefitsQAAgent

    captured: dict = {}

    def _capture(agent_cls, **kwargs):
        captured["agent_cls"] = agent_cls
        captured.update(kwargs)
        return None

    monkeypatch.setattr(marketplace_app, "run_agent_marketplace", _capture)
    runpy.run_path(str(_CLI), run_name="__main__")

    assert captured, "cli.py did not call the Marketplace runner"
    assert captured["agent_cls"] is HRPolicyBenefitsQAAgent, "the runner was handed the wrong graph class"
    assert captured["agent_name"] == "HRPolicyBenefitsQAAgent"
    assert captured["namespace"] == "svc"

    declared = _config_yaml()
    assert declared, "config/config.yaml is empty — nothing to verify"
    for key, value in declared.items():
        assert captured["config"].get(key) == value, f"{key} never reached the runner"
