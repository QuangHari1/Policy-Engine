from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from langfuse import get_client
from langfuse.openai import OpenAI
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str | None
    api_key: str | None
    model: str
    timeout: float = 30.0


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=0,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="span",
            name="llm-chat",
            input={
                "model": self.config.model,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
            },
        ) as span:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            span.update(output={"response_length": len(content)})
            return content

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        response = self.client.beta.chat.completions.parse(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
            temperature=0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no structured output")
        return parsed
