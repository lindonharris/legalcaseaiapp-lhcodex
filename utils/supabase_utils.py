import os
import json
import logging
import uuid
import boto3
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_client: Client = create_client(
    os.getenv('SUPABASE_PROJECT_URL'), 
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')  # Ensure this key is the service role
)

# Init logger
logger = logging.getLogger(__name__)

def create_new_chat_session(
    client, 
    table_name,
    user_id,
    project_id
):
    '''INSERT/CREATE a new chat_session object table public.chat_sessions'''
    try:
        response = client.table(table_name).insert({
            "user_id": user_id,
            "project_id": project_id,
        }).execute()
    except Exception as e:
        raise Exception(f"Error saving to public.chat_sessions: {e}")

def insert_mp3_supabase_record(
    client, 
    table_name, 
    podcast_title, 
    cdn_url, 
    transcript,
    content_tags,
    uploaded_by, 
    is_public,
    is_playlist,
):
    """Inserts a record into the Supabase Library table."""
    try:
        data = {
            "podcast_title": podcast_title,
            "cdn_url": cdn_url,
            "uploaded_by":uploaded_by,
            "is_public": is_public,
            "transcript": transcript,
            "description": 'this description could be workshopped just a bit...',
            "is_playlist": False
        }
        # Execute the insert query
        response = client.table(table_name).insert(data).execute()   
        
        # Check if the response contains data
        if response.data:
            # Successful insertion
            print("Successful Supabase 'podcast' row insert")
            return response.data[0]["id"]  # Return podcast ID... prior version is just response.data
        else:
            # If there's no data, check for errors
            raise Exception(f"Supabase error: {response.error}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase: {e}")

def insert_document_supabase_record(client, table_name, cdn_url, project_id, content_tags, uploaded_by):
    """Inserts a record into the Supabase Library table."""
    try:
        data = {
            "cdn_url": cdn_url,
            "project_id": project_id,
            "content_tags": content_tags,
            "uploaded_by":uploaded_by,
        }
        # Execute the insert query
        response = client.table(table_name).insert(data).execute()   
        
        # Check if the response contains data
        if response.data:
            # Successful insertion
            print("Successful Supabase row insert")
            return response.data[0]["id"]  # Return document_sources.id
        else:
            # If there's no data, check for errors
            raise Exception(f"Supabase error: {response.error}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase: {e}")

def insert_vector_supabase_record(client, table_name, source_id, project_id, content, metadata, embedding):
    '''INSERT into table public.document_vector_store'''
    try:
        response = client.table(table_name).insert({
            "source_id": source_id,
            "content": content,
            "metadata": json.dumps(metadata),
            "embedding": embedding,
            "project_id": project_id
        }).execute()
        logger.info(f"Response status={response.status} and statusText={response.statusText}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase vector-store: {e}")

def insert_note_supabase_record(
    client, 
    table_name, 
    user_id, 
    project_id, 
    content_markdown, 
    note_type, 
    is_shareable, 
    created_at,
):
    '''INSERT note into table public.note'''
    try:
        response = client.table(table_name).insert({
            "user_id": user_id,
            "project_id": project_id,
            "note_type": note_type,
            "content_markdown": content_markdown,
            "created_at": created_at,
            "is_shareable": is_shareable
        }).execute()
    except Exception as e:
        raise Exception(f"Error saving to public.note: {e}")

def insert_chat_message_supabase_record(
    client, 
    table_name, 
    user_id, 
    chat_session_id, 
    dialogue_role, 
    message_content, 
    query_response_status,
    created_at,
):
    '''INSERT message into table public.messages'''
    try:
        response = client.table(table_name).insert({
            "user_id": user_id,
            "chat_session_id": chat_session_id,
            "role": dialogue_role,                              # enum(user, assistant)
            "content": message_content,
            "query_response_status": query_response_status,     # status for SUCCESS or FAILED llm query response
            "raw_resposne": "",
            "created_at": created_at,
        }).execute()
        return response
    except Exception as e:
        raise Exception(f"Error saving to public.messages: {e}")