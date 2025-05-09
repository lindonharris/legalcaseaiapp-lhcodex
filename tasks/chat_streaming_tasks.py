'''
This file runs Celery tasks for handling RAG chat tasks with token streaming enabled
'''

from celery import Celery, Task
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
from supabase import create_client, Client
from utils.supabase_utils import insert_conversation_supabase_record, supabase_client
from datetime import datetime, timezone
import requests
import tempfile
import uuid
import json

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

logger = logging.getLogger(__name__)

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

def get_chat_llm(model_name: str, callback_manager: CallbackManager = None) -> ChatOpenAI:
    """
    Factory to create a ChatOpenAI instance.
    If callback_manager is provided, enable streaming and attach handlers.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=0.7,
        streaming=(callback_manager is not None),  # toggle streaming if callbacks exist
        callback_manager=callback_manager        # attaches our StreamToClientHandler
    )

@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_chat_streaming_task(self, user_id, chat_session_id, query, project_id, model_type):
    """
    Main RAG workflow:
    1. Ensure a chat session exists
    2. Embed the user query
    3. Retrieve relevant document chunks
    4. Stream LLM response tokens to client and collect full answer
    5. Persist query and answer in Supabase
    """
    try:
        # 1) Initialize or create conversation if needed
        if not chat_session_id:
            chat_session_id = create_new_conversation(user_id, project_id)

        # 2) Embed the query using OpenAI Ada embeddings (1536 dims)
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        query_embedding = embedding_model.embed_query(query)

        # 3) Fetch top-K relevant chunks via Supabase RPC
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)

        # 4) Stream generation to client, accumulate full answer
        full_answer = generate_rag_answer(
            query,
            chat_session_id,
            relevant_chunks,
            model_name=model_type  # supports gpt-4o, gemini-flash, deepseek-v3, etc.
        )

        # 5) Save entire conversation turn in DB
        save_conversation(chat_session_id, user_id, query, answer=full_answer)

        # Return completed answer for HTTP response
        return {"answer": full_answer, "message_role": "assistant"}

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
    full_context = trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens=127999)

    # Set up streaming callback for token-level pushes
    handler = StreamToClientHandler(chat_session_id)
    cb_manager = CallbackManager([handler])
    llm = get_chat_llm(model_name, callback_manager=cb_manager)

    # Streaming prediction: on_llm_new_token fires for each token
    answer = llm.predict(full_context)
    return answer

def create_new_conversation(user_id, document_ids):
    """
    Inserts a new row into chat_session and returns its UUID.
    """
    new_chat_session = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_ids": document_ids
    }
    response = supabase_client.table("chat_session").insert(new_chat_session).execute()
    return response.data[0]["id"]

def save_conversation(chat_session_id, user_id, query, answer):
    """
    Persists user and assistant messages into Supabase messages table.
    """
    insert_conversation_supabase_record(
        supabase_client,
        table_name="message",
        user_id=user_id,
        chat_session_id=chat_session_id,
        message_role="user",
        message_content=query,
        created_at=datetime.now(timezone.utc).isoformat()
    )
    insert_conversation_supabase_record(
        supabase_client,
        table_name="message",
        user_id=user_id,
        chat_session_id=chat_session_id,
        message_role="assistant",
        message_content=answer,
        created_at=datetime.now(timezone.utc).isoformat()
    )

def fetch_chat_history(chat_session_id):
    """
    Returns ordered list of message dicts for a conversation.
    """
    response = supabase_client.table("message").select("*").eq("chat_session_id", chat_session_id).order("created_at").execute()
    return response.data

def format_chat_history(chat_history):
    """
    Converts list of messages to a single string for prompt context.
    """
    return "".join(f"{m['message_role'].capitalize()}: {m['message_content']}\n" for m in chat_history)

def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    """
    Iteratively remove chunks until prompt fits within model token limit.
    """
    import tiktoken
    tokenizer = tiktoken.encoding_for_model(model_name)
    history = full_context
    # While over token budget, drop least relevant chunks
    while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
        relevant_chunks.pop()
        chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
        history = f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
    return full_context