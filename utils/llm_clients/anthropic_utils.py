# utils/llm_clients/anthropic_client.py

import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Anthropic API key not found")

class AnthropicClient:
    def __init__(self, model_name: str = "claude-4", temperature: float = 0.7):
        self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model_name
        self.temperature = temperature

    def chat(self, prompt: str) -> str:
        """
        Anthropic’s convention: we wrap user prompt with HUMAN_PROMPT and then AI_PROMPT.
        """
        full_prompt = f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
        response = self._client.completions.create(
            model=self.model_name,
            prompt=full_prompt,
            temperature=self.temperature,
            max_tokens=4096,
        )
        return response.completion  # the assistant’s reply

