'''
This file runs Celery tasks for handling RAG AI note creation tasks (outlines, summaries, compare-contrast)
Note genereation (with RAG) is done without token streaming. Returns full answer in one go.
'''

from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
import redis
from supabase import create_client, Client
from utils.supabase_utils import insert_note_supabase_record, supabase_client
from datetime import datetime, timezone
import tiktoken
import requests
import tempfile
import uuid
import json
from dotenv import load_dotenv

# langchain imports
from langchain_core.load import dumpd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#### GLOBAL ####
# API Keys
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_PROD_KEY")

# Define which models support temp (and their allowed ranges, if you like)
_MODEL_TEMPERATURE_CONFIG = {
    # no-temperature models
    "o4-mini":      {"supports_temperature": False},
    "gpt-4o-mini":  {"supports_temperature": False},
    # temperature-capable models
    "gpt-4.1-nano": {"supports_temperature": True, "min": 0.0, "max": 2.0},
}

# Logging
logger = logging.getLogger(__name__)

# Initialize Redis sync client for pub/sub
REDIS_LABS_URL = 'redis://default:' + os.getenv("REDIS_PASSWORD") + '@' + os.getenv("REDIS_PUBLIC_ENDPOINT")
redis_sync = redis.Redis.from_url(REDIS_LABS_URL, decode_responses=True)

class BaseTaskWithRetry(Task):
    """
    Base Celery Task class enabling automatic retries on exceptions.
    """
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": 5}
    retry_jitter = True

def publish_token(chat_session_id: str, token: str):
    """
    Stub function to publish a single token to clients subscribed to a chat session.
    Replace with actual pub/sub (e.g. Redis, Supabase Realtime).
    """
    # Example: redis.publish(f"chat:{chat_session_id}", token)
    pass

class StreamToClientHandler(BaseCallbackHandler):
    """
    LangChain callback handler that emits each new LLM token via publish_token().
    """
    def __init__(self, session_id: str):
        self.session_id = session_id

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Called by LangChain for every new token in streaming mode
        publish_token(self.session_id, token)

def get_chat_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    callback_manager: any = None,
) -> "ChatOpenAI":
    """
    Factory to create a ChatOpenAI instance.
    If callback_manager is provided, enable streaming and attach handlers.

    (default) OpenAI o4-mini model
    """
    from langchain_openai import ChatOpenAI     # Lazy-import to reduce startup footprint

    cfg = _MODEL_TEMPERATURE_CONFIG.get(model_name, {"supports_temperature": True})
    llm_kwargs = {
        "api_key": OPENAI_API_KEY,
        "model": model_name,
        "streaming": False,     # Disable streaming by default
    }
    # Only include temperature if this model supports it
    if cfg.get("supports_temperature", False):
        lo, hi = cfg.get("min", 0.0), cfg.get("max", 2.0)
        safe_temp = max(lo, min(temperature, hi))
        llm_kwargs["temperature"] = safe_temp

    # Attach streaming callback if provided
    if callback_manager:
        llm_kwargs["streaming"] = True
        llm_kwargs["callback_manager"] = callback_manager

    return ChatOpenAI(**llm_kwargs)

# def get_chat_llm(
#         model_name: str = "gpt-4o-mini", 
#         callback_manager: CallbackManager = None
# ) -> ChatOpenAI:
#     """
#     Factory to create a ChatOpenAI instance.
#     If callback_manager is provided, enable streaming and attach handlers.

#     (default) OpenAI gpt-4o-mini model
#     """
#     return ChatOpenAI(
#         model=model_name,
#         api_key=OPENAI_API_KEY,         # ← supply your key here
#         temperature=0.7,
#         streaming=False,                # Disable streaming if callbacks exist
#         callback_manager=None           # attaches our StreamToClientHandler
#     )

@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_note_task(
    self, 
    user_id, 
    note_type, 
    project_id, 
    model_name
):
    """
    Main RAG workflow:
    1. Embed the generate note query as dim-1536
    2. Retrieve relevant document chunks
    3. Stream LLM response tokens to client and collect full answer
    4. Persist query and answer in Supabase
    """
    try:
        # Set explicit start time in self.metadata
        # This manually sets result = AsyncResult(task_id) when checking on this celery task via job.id
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        self.update_state(
            state="STARTED", 
            meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        if note_type == "outline":
            query = "Create a comprehensive oputline of the following documents"

        if note_type == "exam-questions":
            query = "Based on the documents (appended as rag context) create an list of 15 exam questions"

        # Step 1) Embed the query using OpenAI Ada embeddings (1536 dims)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY)
        query_embedding = embedding_model.embed_query(query)

        # Step 2) Fetch top-K relevant chunks via Supabase RPC
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)

        # Step 3) Generation answer for client (@TODO may be used to token streaming at a later point )
        full_answer = generate_rag_answer(
            query,
            relevant_chunks,
            model_name=model_name  # supports gpt-4o, gemini-flash, deepseek-v3, etc.
        )

        # Step 5) Save note to the public.notes table in Supabase (realtime Supabase table)
        save_note(
            project_id,
            user_id, 
            note_type, 
            content=full_answer      # full rag reponse
        )

        # Return nothing
        return "RAG Note Task suceess"

    except Exception as e:
        logger.error(f"PDF processing failed: {e}", exc_info=True)
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            raise RuntimeError(
                f"[CELERY] Step 2) RAG Note creattion failed permanently after {self.max_retries} retries: {e}"
            ) from e

def fetch_relevant_chunks(query_embedding, project_id, match_count=10):
    """
    Calls Supabase RPC to retrieve the nearest neighbor chunks using HNSW index.
    """
    try:
        response = supabase_client.rpc("match_document_chunks_hnsw", {
            "p_project_id": project_id,
            "p_query": query_embedding,
            "p_k": match_count
        }).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching relevant chunks: {e}", exc_info=True)
        raise

def generate_rag_answer(query, relevant_chunks, model_name, max_chat_history=10):
    """
    Build prompt, invoke streaming LLM, publish tokens in real-time,
    and return the full generated answer at completion.
    """
    # Build conversational context
    # chat_history = fetch_chat_history(chat_session_id)[-max_chat_history:]
    # formatted_history = format_chat_history(chat_history) if chat_history else ""
    chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
    full_context = (
        f"Relevant Context:\n{chunk_context}\n\n"
        f"User Query: {query}\nAssistant:"
    )
    full_context = trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens=127999)

    # Set up streaming callback for token-level pushes
    # handler = StreamToClientHandler(chat_session_id)                     # Disabled, no token streaming
    # cb_manager = CallbackManager([handler])                              # Disabled, no token streaming
    llm = get_chat_llm(model_name)

    # Streaming prediction: on_llm_new_token fires for each token
    answer = llm.predict(full_context)
    return answer

def create_new_conversation(user_id, project_id):
    """
    Inserts a new row into chat_session and returns its UUID... for now not in use (using WeWeb to insert a new convo)
    """
    new_chat_session = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id
    }
    response = supabase_client.table("chat_session").insert(new_chat_session).execute()
    return response.data[0]["id"]

def restart_conversation(user_id: str, project_id: str) -> str:
    """
    Creates a new chat session in public.chat_sessions, updates the 
    chat_session_id in public.projects, then deletes the old session.
    
    Returns the new chat_session_id.
    """
    # 1) Fetch the old session id for this project
    proj_res = (
        supabase_client
        .table("projects")
        .select("chat_session_id")
        .eq("id", project_id)
        .single()
        .execute()
    )
    if proj_res.error:
        raise RuntimeError(f"Error fetching project: {proj_res.error.message}")
    
    old_session_id = proj_res.data["chat_session_id"]
    
    # 2) Insert a new chat_sessions row
    new_chat_session = {
        "user_id": user_id,
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    insert_res = (
        supabase_client
        .table("chat_sessions")
        .insert(new_chat_session)
        .execute()
    )
    if insert_res.error:
        raise RuntimeError(f"Error creating chat session: {insert_res.error.message}")
    
    # Supabase returns a list of inserted rows
    new_session_id = insert_res.data[0]["id"]
    
    # 3) Update the project to point to the new session
    update_res = (
        supabase_client
        .table("projects")
        .update({"chat_session_id": new_session_id})
        .eq("id", project_id)
        .execute()
    )
    if update_res.error:
        raise RuntimeError(f"Error updating project: {update_res.error.message}")
    
    # 4) Delete the old chat session
    #    (only if it existed in the first place)
    if old_session_id:
        delete_res = (
            supabase_client
            .table("chat_sessions")
            .delete()
            .eq("id", old_session_id)
            .execute()
        )
        if delete_res.error:
            raise RuntimeError(f"Error deleting old session: {delete_res.error.message}")
    
    return new_session_id


def save_note(
        project_id, 
        user_id, 
        note_type, 
        content
    ):
    """
    Persists generated summary note into Supabase public.notes table.
    """
    insert_note_supabase_record(
        client=supabase_client,
        table_name="notes",
        user_id=user_id,
        project_id=project_id,
        content_markdown=content,
        note_type=note_type,
        is_shareable=False,
        created_at=datetime.now(timezone.utc).isoformat()
    )

# def fetch_chat_history(chat_session_id):
#     """
#     Returns ordered list of message dicts for a conversation.
#     """
#     response = supabase_client.table("messages").select("*").eq("chat_session_id", chat_session_id).order("created_at").execute()
#     return response.data

# def format_chat_history(chat_history):
#     """
#     Converts list of messages to a single string for prompt context.
#     """
#     return "".join(f"{m['message_role'].capitalize()}: {m['message_content']}\n" for m in chat_history)

def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    model_to_encoding = {
        "o4-mini": "o200k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }
    encoding_name = model_to_encoding.get(model_name, "cl100k_base")
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    history = full_context
    while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
        relevant_chunks.pop()
        chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
        history = f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
    return history