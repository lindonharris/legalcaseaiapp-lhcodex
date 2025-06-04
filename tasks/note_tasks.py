'''
This file runs Celery tasks for handling RAG AI note creation tasks (outlines, summaries, compare-contrast)
Note genereation (with RAG) is done without token streaming. Returns full answer in one go.
'''
import multiprocessing as mp
import traceback
from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError, TimeoutError
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
import redis
from supabase import create_client, Client
from utils.supabase_utils import insert_note_supabase_record, supabase_client
from utils.prompt_utils import load_yaml_prompt, build_prompt_template_from_yaml
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


@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_note_task(
    self, 
    user_id, 
    note_type, 
    project_id, 
    provider: str,  # ← “openai”, “anthropic”, etc.
    model_name: str,
    temperature: float = 0.7,
    metadata: dict = None,      # <-- a dict containing optional overrides, like {"num_questions": 10}
):
    """
    RAG workflow for various note types (exam questions, case briefs, outlines, etc.).

    Steps:
    1) Embed a short “retrieval” query (as dim-1536 embedding)
    2) Fetch top-K relevant chunks via Supabase RPC
    3) Load the appropriate YAML prompt based on note_type
    4) Format that YAML template with {context} + any override from meta (e.g. num_questions)
    5) Call the LLM once with the fully formed prompt (Stream LLM response tokens or return all at once)
    6) Persist note output to public.notes
    """
    
    if metadata is None:
        metadata = {}
    
    try:
        # Step 0) Mark explicit start time in self.metadata
        # This manually sets result = AsyncResult(task_id) when checking on this celery task via job.id
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        self.update_state(
            state="STARTED", 
            meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        # Step 1) Choose the prompt based on user selection of "note_type"
        if note_type == "outline":
            query = "Create a comprehensive outline of the following documents"
            yaml_file = "case-outline-prompt.yaml"
        elif note_type == "exam_questions":
            query = "Based on the documents (appended as rag context) create an list of 15 exam questions"
            yaml_file = "exam-questions-prompt.yaml"
        elif note_type == "case_brief":
            query = "Based on the documents create a comprehensive case brief"
        elif note_type == "compare_contrast":
            query = "Based on the documents create a compare and contrast of the cases"
        else:
            raise ValueError(f"Unknown note_type: {note_type}")
        
        # Load that YAML
        yaml_dict = load_yaml_prompt(yaml_file)
        prompt_template = build_prompt_template_from_yaml(yaml_dict)

        # Step 2) WHAT SHOULD GO HERE?? Embed the query using OpenAI Ada embeddings (1536 dims)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY)
        query_embedding = embedding_model.embed_query(query)

        # Step 3) Fetch top-K relevant chunks via Supabase RPC
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)

        # Step 4) Build the final LLM prompt (YAML + Embeddings + Num Questions)
        chunk_context = "\n\n".join(chunk["content"] for chunk in relevant_chunks)

        # Determine num_questions (if from user via metadata["num_questions"] or using YAML fallback
        num_questions = metadata.get("num_questions")
        if note_type == "exam_questions":
            # If YAML used a placeholder named “n_questions”, we must supply it.
            if num_questions and isinstance(num_questions, int) and num_questions > 0:
                llm_input = prompt_template.format(
                    context=chunk_context,
                    n_questions=num_questions
                )
            else:
                # No override; assume the YAML has a default. If you want a fallback,
                # you could do: llm_input = prompt_template.format(context=chunk_context, n_questions=15)
                llm_input = prompt_template.format(
                    context=chunk_context,
                    n_questions=metadata.get("num_questions", 15)
                )
        elif note_type == "case_brief":
            # Case brief template only has {context} placeholder.
            llm_input = prompt_template.format(context=chunk_context)

        # Step 5) Generate llm client from factory
        from utils.llm_factory import LLMFactory       # Lazy-import heavy modules
        llm_client = LLMFactory.get_client(
            provider=provider,
            model_name=model_name,
            temperature=temperature
        )

        # Step 6) Generate answer for client
        # full_answer = generate_rag_answer(
        #     llm_client=llm_client,
        #     query=query,
        #     relevant_chunks=relevant_chunks
        # )
        full_answer = llm_client.chat(llm_input)

        # Step 7) Save note to the public.notes table in Supabase (realtime Supabase table)
        save_note(
            project_id,
            user_id, 
            note_type, 
            content=full_answer      # full rag reponse
        )

        # Return nothing
        return "RAG Note Task suceess"

    except Exception as e:
        logger.error(f"RAG Note Task failed: {e}", exc_info=True)
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

def generate_rag_answer(
        llm_client,
        query, 
        relevant_chunks: list,
        max_chat_history: int = 10,
    ):
    """
    [NOTE_TASK SPECIFIC[]
    Build prompt, invoke streaming LLM, publish tokens in real-time,
    and return the full generated answer at completion.

    Q: **Why no `chat_session_id` ?
    A: because this is going to a project not a chat histoty
    """
    # Build conversational context
    # chat_history = fetch_chat_history(chat_session_id)[-max_chat_history:]
    # formatted_history = format_chat_history(chat_history) if chat_history else ""
    chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
    full_context = (
        f"Relevant Context:\n{chunk_context}\n\n"
        f"User Query: {query}\nAssistant:"
    )

    trimmed_context = trim_context_length(
        full_context=full_context,
        query=query,
        relevant_chunks=relevant_chunks,
        model_name=llm_client.model_name,
        max_tokens=127999
    )

    # 5) Finally, call the LLM client once
    try:
        # All of your LLMClient implementations expose `.chat(prompt: str) -> str`
        answer = llm_client.chat(trimmed_context)
        # llm_client = get_chat_llm(model_name) # being SUNSET !!! 
        return answer
    except Exception as e:
        logger.error(f"Error in LLM call (model={llm_client.model_name}): {e}", exc_info=True)
        raise

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