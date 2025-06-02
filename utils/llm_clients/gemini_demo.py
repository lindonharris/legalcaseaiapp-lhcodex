import os
import json
import vertexai
from google.oauth2 import service_account
from dotenv import load_dotenv

from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = "jurisai-461700"
LOCATION = "us-central1"

# Load environment variables
load_dotenv()

# --- Authentication and Vertex AI Initialization ---
credentials_json_string = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if credentials_json_string:
    try:
        credentials_info = json.loads(credentials_json_string)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        print("Vertex AI initialized using service account JSON from env.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}. Falling back to default credentials.")
        import google.auth
        credentials, project = google.auth.default()
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        print("Vertex AI initialized using Application Default Credentials.")
else:
    import google.auth
    credentials, project = google.auth.default()
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    print("Vertex AI initialized using Application Default Credentials (no JSON env var found).")

# --- API Test Script for Vertex AI Gemini ---
model_name = "gemini-2.5-flash-preview-05-20" # Or "gemini-1.5-flash-001", etc.

try:
    model = GenerativeModel(model_name)
    response = model.generate_content("How does AI work?")
    
    print(f"\n--- Response from {model_name} ---")
    print(response.text)

except Exception as e:
    print(f"An error occurred during API call: {e}")
    print(f"Ensure model '{model_name}' is available in '{LOCATION}', service account roles are correct, and Vertex AI API is enabled.")