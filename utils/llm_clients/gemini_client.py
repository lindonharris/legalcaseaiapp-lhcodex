# utils/llm_clients/gemini_client.py

import os
import json
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel # Corrected import
from google.oauth2 import service_account

load_dotenv()

GOOGLE_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID", "").strip()
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1").strip()
CREDENTIALS_JSON_STR = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")

if not GOOGLE_PROJECT_ID:
    raise ValueError("GEMINI_PROJECT_ID not set")

if CREDENTIALS_JSON_STR:
    info = json.loads(CREDENTIALS_JSON_STR)
    credentials = service_account.Credentials.from_service_account_info(info)
    vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION, credentials=credentials)
else:
    vertexai.init(project=GOOGLE_PROJECT_ID, location=GEMINI_LOCATION)

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20", temperature: float = 0.7, max_output_tokens: int = 512):
        """
        model_name must match one of the “pretrained” Gemini model IDs your project has access to.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def chat(self, prompt: str) -> str:
        try:
            # Use GenerativeModel directly
            model = GenerativeModel(self.model_name) # Changed from TextGenerationModel.from_pretrained
            response = model.generate_content( # Changed from model.predict
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            # Access the text from the response object
            return response.candidates[0].content.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error calling `{self.model_name}`: {e}")