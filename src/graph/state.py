from typing import Any
from pydantic import BaseModel, Field


class PolicyState(BaseModel):
    business_requirement: dict[str, Any]
    detected_services: list[str] = Field(default_factory=list)
    service_outputs: dict[str, Any] = Field(default_factory=dict)
    final_configuration: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)