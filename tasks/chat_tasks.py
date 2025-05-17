'''
This file runs Celery tasks for handling RAG chat tasks (non-streaming LLM responses)
Handles RAG chat tasks without token streaming. Returns full answer in one go.

Models
o4-mini
gpt-4.1-nano

'''

from celery import Celery, Task
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
import redis
from supabase import create_client, Client
from utils.supabase_utils import supabase_client, insert_chat_message_supabase_record, create_new_chat_session
from datetime import datetime, timezone
import requests
import tempfile
import uuid
import json
import tiktoken
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

# API Keys
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_PROD_KEY")

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

def get_chat_llm(model_name: str = "o4-mini", callback_manager: CallbackManager = None) -> ChatOpenAI:
    """
    Factory to create a ChatOpenAI instance.
    If callback_manager is provided, enable streaming and attach handlers.

    (default) OpenAI o4-mini model
    """
    return ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        streaming=False,                            # Disable streaming if callbacks exist
        callback_manager=None                       # attaches our StreamToClientHandler
    )

@celery_app.task(bind=True, base=BaseTaskWithRetry)
def new_chat_session(self, user_id, project_id):
    """
    Simple task to creae a new chat_session and associate it query to public.messages in Supabase before performing RAG Q&A

    1) INSERT a new blank conversation row into Supabase public.chat_sessions
    2) UPDATE a given public.project's newly created "chat_session_id" column
    """
    try:
        # Step 1) Create blank chat_session object if needed
        response = create_new_chat_session(
            supabase_client,
            table_name="chat_sessions",
            user_id=user_id, 
            project_id=project_id,
        )
        new_chat_session_id = response.data[0]["id"]

        # Step 2) UPDATE/REPLACE public.projects, chat_session_id column
        # @TODO: I WILL have orphan chats in chat_sessions that need to be cleared out when
        # the user clears the chat 🔃 session in the UI
        try:
            _ = supabase_client \
            .table("projects") \
            .update({"chat_session_id": new_chat_session_id}) \
            .eq("id", project_id) \
            .execute()
        except Exception as e:
            raise Exception(f"Error saving to public.projects: {e}")
        return None

    except Exception as e:
        logger.error(f"Create new_chat_session task failed: {e}", exc_info=True)
        # Retry on failure according to BaseTaskWithRetry policies
        raise self.retry(exc=e)

@celery_app.task(bind=True, base=BaseTaskWithRetry)
def persist_user_query(self, user_id, chat_session_id, query, project_id, model_type):
    """
    Simple task to persist user query to public.messages in Supabase before performing RAG Q&A
    """
    try:
        # Set explicit start time metadata
        # This manually sets result = AsyncResult(task_id) when checking on this celery task via job.id
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        self.update_state(
            state="STARTED", # ← using "PENDING" can confuse clients into thinking the task has not started yet
            meta={
                "start_time": datetime.now(timezone.utc).isoformat()
            }
        )

        # Step 1) Persist user query to public.messages
        response = insert_chat_message_supabase_record(
            supabase_client,
            table_name="messages",
            user_id=user_id,
            chat_session_id=chat_session_id,
            dialogue_role="user",
            message_content=query,
            query_response_status="PENDING",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Step 2) Return completed answer for HTTP response (unpacked in WeWeb)
        inserted_id = response.data[0]["id"]
        return inserted_id

    except Exception as e:
        logger.error(f"RAG Chat Task failed: {e}", exc_info=True)
        # Retry on failure according to BaseTaskWithRetry policies
        raise self.retry(exc=e)

@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_chat_task(
    self, 
    message_id,     # ← note: passed in first by chained task
    user_id, 
    chat_session_id, 
    query, 
    project_id, 
    model_type,
):
    """
    Main RAG workflow, implementing the standard “optimistic UI” pattern
    1. Embed the user query
    2. Fetch top-K relevant chunks
    3. Generate llm_assistant answer
    4. Persist llm assistant answer in Supabase public.messages table (retry failure precaution?)
    """
    try:
        # Set explicit start time metadata
        # This manually sets result = AsyncResult(task_id) when checking on this celery task via job.id
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        self.update_state(
            state="STARTED", # ← using "PENDING" can confuse clients into thinking the task has not started yet
            meta={
                "start_time": datetime.now(timezone.utc).isoformat()
            }
        )

        if model_type == "":
            model_type="o4-mini"

        # Step 1) Embed the query using OpenAI Ada embeddings (1536 dims)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
        )
        query_embedding = embedding_model.embed_query(query)

        # Step 2) Fetch top-K relevant chunks via Supabase RPC
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)

        # Step 3) Generation answer for client (@TODO may be used to token streaming at a later point )
        assistant_response = generate_rag_answer(
            query,
            chat_session_id,
            relevant_chunks,
            model_name=model_type  # supports gpt-4o, gemini-flash, deepseek-v3, etc.
        )

        # Step 4) insert assistant response
        _ = insert_chat_message_supabase_record(
            supabase_client,
            table_name="messages",
            user_id=user_id,
            chat_session_id=chat_session_id,
            dialogue_role="assistant",
            message_content=assistant_response,
            query_response_status="COMPLETE",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Step 5) UPDATE public.messages column ???????
        try:
            _ = supabase_client \
            .table("messages") \
            .update({"query_response_status": "COMPLETE"}) \
            .eq("id", message_id) \
            .execute()
        except Exception as e:
            raise Exception(f"Error updating public.messages status: {e}")
        return None

    except Exception as e:
        logger.error(f"RAG Chat Task failed: {e}", exc_info=True)
        # Retry on failure according to BaseTaskWithRetry policies
        raise self.retry(exc=e)

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

def generate_rag_answer(query, chat_session_id, relevant_chunks, model_name, max_chat_history=10):
    """
    Build prompt, invoke streaming LLM, publish tokens in real-time,
    and return the full generated answer at completion.
    """
    # Build conversational context
    chat_history = fetch_chat_history(chat_session_id)[-max_chat_history:]
    formatted_history = format_chat_history(chat_history) if chat_history else ""
    chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
    full_context = (
        f"{formatted_history}\n\nRelevant Context:\n{chunk_context}\n\n"
        f"User Query: {query}\nAssistant:"
    )
    trimmed_context  = trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens=127999)

    # Set up streaming callback for token-level pushes
    # handler = StreamToClientHandler(chat_session_id)                     # Disabled, no token streaming
    # cb_manager = CallbackManager([handler])                              # Disabled, no token streaming
    llm = get_chat_llm(model_name)

    # Streaming prediction: on_llm_new_token fires for each token
    answer = llm.predict(trimmed_context )
    return answer

def create_new_conversation(user_id, project_id):
    """
    Inserts a new row into chat_session and returns its UUID.
    """
    new_chat_session = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id
    }
    response = supabase_client.table("chat_session").insert(new_chat_session).execute()
    return response.data[0]["id"]

def save_conversation(chat_session_id, user_id, query, answer):
    """
    Persists user and assistant messages into Supabase public.messages table.
    """
    # # insert user query
    # insert_chat_message_supabase_record(
    #     supabase_client,
    #     table_name="message",
    #     user_id=user_id,
    #     chat_session_id=chat_session_id,
    #     dialogue_role="user",
    #     message_content=query,
    #     created_at=datetime.now(timezone.utc).isoformat()
    # )
    # insert assistant response
    insert_chat_message_supabase_record(
        supabase_client,
        table_name="messages",
        user_id=user_id,
        chat_session_id=chat_session_id,
        dialogue_role="assistant",
        message_content=answer,
        query_response_status="",
        created_at=datetime.now(timezone.utc).isoformat()
    )

def fetch_chat_history(chat_session_id):
    """
    Returns ordered list of message dicts for a conversation.
    """
    response = supabase_client.table("messages") \
        .select("*") \
        .eq("chat_session_id", chat_session_id) \
        .order("created_at").execute()
    return response.data

def format_chat_history(chat_history):
    """
    Converts list of messages to a single string for prompt context.
    """
    formatted_string = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in chat_history)
    return formatted_string.strip()

def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    """
    Iteratively remove chunks until prompt fits within model token limit.
    """
    # Map custom/internal model names to known tiktoken encoding names
    # Add mappings for all models you expect to use here
    model_to_encoding = {
        "o4-mini": "o200k_base", # Assuming o4-mini uses the gpt-4o encoding
        "gpt-4o": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        # Add other mappings if you use different models (e.g., for Claude, etc.)
        # Note: Non-OpenAI models might require different tokenization libraries or approaches.
        # This mapping works ONLY for models using tiktoken encodings.
    }

    # Get the encoding name based on the model_name.
    # Provide a fallback (like 'cl100k_base') or raise an error if the model name is not in the map.
    encoding_name = model_to_encoding.get(model_name)

    if not encoding_name:
        logger.warning(f"Unknown model name '{model_name}'. Cannot determine tiktoken encoding. Using fallback 'cl100k_base'.")
        encoding_name = "cl100k_base" # Fallback to a common encoding

    try:
        # Use get_encoding with the determined encoding name
        tokenizer = tiktoken.get_encoding(encoding_name)
    except ValueError:
        logger.error(f"Failed to get tiktoken encoding for name '{encoding_name}' (mapped from model '{model_name}'). Falling back to 'cl100k_base'.")
        # Fallback if the mapped encoding name is itself invalid
        tokenizer = tiktoken.get_encoding("cl100k_base")


    history = full_context # Initialize token calculation with the full context string

    # Note: The original loop logic here modifies relevant_chunks and recalculates 'history'
    # based on chunks and query. However, the function *returns* the original
    # 'full_context' string, making the trimming ineffective as written in the original code.
    # This fix addresses the tiktoken error but does not correct this potential logic bug
    # in how the trimmed context is returned.
    while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
        relevant_chunks.pop() # Remove the least relevant chunk

        # Rebuild the string used for token counting in the loop.
        chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
        history = f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"

    return history # <-- Change to this to make trimming effective and return the string used in the loop calculation.

# def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens): 
#     """
#     Iteratively remove chunks until prompt fits within model token limit.
#     """
#     import tiktoken
#     try:
#         if model_name == "":
#             model_name="o4-mini"
    
#         tokenizer = tiktoken.encoding_for_model(model_name)
#     except KeyError:
#         logger.warning(f"Unknown model '{model_name}', falling back to 'cl100k_base' tokenizer")
#         tokenizer = tiktoken.get_encoding("cl100k_base")
#     history = full_context
#     # While over token budget, drop least relevant chunks
#     while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
#         relevant_chunks.pop()
#         chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
#         history = f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
#     return full_context
