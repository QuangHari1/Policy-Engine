from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.api.app as app_module
from src.api.app import RouteRequest


def test_route_flow_real_workflow():
    payload = json.loads(Path("data/fake/voice_01.json").read_text(encoding="utf-8"))

    response = app_module.route(RouteRequest(business_requirement=payload))

    result = response.model_dump()
    print(result)
    assert "detected_services" in result
    assert isinstance(result["detected_services"], list)
    assert result["detected_services"]


@pytest.mark.skipif(
    not Path("data/fake/voice_01.json").exists(),
    reason="fake KBKD sample is missing",
)
def test_route_sample_file_exists():
    assert Path("data/fake/voice_01.json").exists()
