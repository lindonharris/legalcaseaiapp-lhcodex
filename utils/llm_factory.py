# Factory/Dispatcher (fetches clients from /llm_clients)

'''
`utils/llm_factory.py`: reads provider + model name + temperature and 
returns an instance of whichever client class you need.
'''

from typing import Any
from utils.llm_clients.openai_client import OpenAIClient
from utils.llm_clients.deepseek_client import DeepSeekClient
from utils.llm_clients.anthropic_client import AnthropicClient
from utils.llm_clients.gemini_client import GeminiClient
# from utils.llm_clients.qwen_client import QWENClient

class LLMFactory:
    """
    Given a provider name + model + temperature, return an object with .chat(prompt) -> str
    """
    @staticmethod
    def get_client(provider: str, model_name: str, temperature: float = 0.7, streaming: bool = False, callback_manager: Any = None):
        provider = provider.lower()
        if provider == "openai":
            return OpenAIClient(model_name=model_name, temperature=temperature, streaming=streaming, callback_manager=callback_manager)
        elif provider == "deepseek":
            return DeepSeekClient(model_name=model_name, temperature=temperature, streaming=streaming, callback_manager=callback_manager)
        elif provider == "anthropic":
            return AnthropicClient(model_name=model_name, temperature=temperature)
        elif provider == "gemini":
            return GeminiClient(model_name=model_name, temperature=temperature)
        elif provider == "qwen":
            return QWENClient(model_name=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Valid options: openai, deepseek, anthropic, gemini, sonnet, qwen.")
