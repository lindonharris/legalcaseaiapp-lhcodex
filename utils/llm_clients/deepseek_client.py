# utils/llm_clients/deepseek_client.py
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Remove the SSL_CERT_FILE variable from the environment if it exists.
os.environ.pop("SSL_CERT_FILE", None)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")  # e.g. "https://api.deepseek.ai/v1"

if not DEEPSEEK_API_KEY or not DEEPSEEK_BASE_URL:
    raise ValueError("DeepSeek credentials or base URL not found in environment")

class DeepSeekClient:
    def __init__(
            self,
            model_name: str = "deepseek-chat", # <- so it's deepseek-chat not deepseek-v3
            temperature: float = 0.7, 
            streaming: bool = False, 
            callback_manager=None
        ):
        # You can either (a) create a raw OpenAI client and call its chat endpoint directly, or
        # (b) wrap it with LangChain.
        llm_kwargs = {
            "api_key": DEEPSEEK_API_KEY,
            "model": model_name,
            "streaming": streaming,
            "base_url": DEEPSEEK_BASE_URL,
        }
        # … (optionally clamp temperature if model supports it) …
        llm_kwargs["temperature"] = temperature

        # Internally use LangChain’s ChatOpenAI but override client
        self._client = ChatOpenAI(**llm_kwargs)

    def chat(self, prompt: str) -> str:
        return self._client.predict(prompt)
