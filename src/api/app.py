from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents.router_agent import RouterAgent
from src.graph.state import PolicyState
from src.graph.workflow import PolicyWorkflow
from src.llm.client import LLMClient, LLMConfig

app = FastAPI(title="Policy Engine", version="0.1.0")
_WORKFLOW: PolicyWorkflow | None = None


class RouteRequest(BaseModel):
    business_requirement: dict[str, Any]


class RouteResponse(BaseModel):
    detected_services: list[str] = Field(default_factory=list)


def build_router_workflow() -> PolicyWorkflow:
    llm_config = LLMConfig(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        timeout=float(os.getenv("OPENAI_TIMEOUT", "30")),
    )
    llm_client = LLMClient(llm_config)
    router_agent = RouterAgent(llm_client)
    return PolicyWorkflow(router_agent=router_agent)


def get_router_workflow() -> PolicyWorkflow:
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = build_router_workflow()
    return _WORKFLOW


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(payload: RouteRequest) -> RouteResponse:
    initial_state = PolicyState(business_requirement=payload.business_requirement)
    result = get_router_workflow().run(initial_state)
    detected_services = (
        result["detected_services"]
        if isinstance(result, dict)
        else result.detected_services
    )
    return RouteResponse(detected_services=detected_services)
