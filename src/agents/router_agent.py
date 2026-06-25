from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.models import RoutingResult
from src.llm.client import LLMClient


class RouterAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_path = Path("src/llm/promts/router.txt")

    def route(self, business_requirement: dict[str, Any]) -> RoutingResult:
        system_prompt = self.prompt_path.read_text(encoding="utf-8")
        user_prompt = json.dumps(business_requirement, ensure_ascii=False, indent=2)
        raw_output = self.llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return RoutingResult.model_validate_json(raw_output)
