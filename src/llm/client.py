from __future__ import annotations

from dataclasses import dataclass
from openai import OpenAI


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
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        return content or ""
