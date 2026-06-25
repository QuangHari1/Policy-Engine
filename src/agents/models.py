from pydantic import BaseModel, Field


class RoutingResult(BaseModel):
    detected_services: list[str] = Field(default_factory=list)
