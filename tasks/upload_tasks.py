"""
This file runs Celery tasks, place any logic that needs to run async 
like api calls and calls to other services. 
"""

from celery import Celery, chain
import logging
import os
import json
import psutil
import requests
import tempfile
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
from utils.audio_utils import generate_audio, generate_only_dialogue_text
from utils.s3_utils import upload_to_s3, generate_presigned_url, s3_client, s3_bucket_name
from utils.supabase_utils import insert_document_supabase_record, insert_mp3_supabase_record, insert_vector_supabase_record, supabase_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.instruction_templates import INSTRUCTION_TEMPLATES
from time import sleep
from datetime import datetime, timezone
import uuid

# langchain dependencies
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

# === PRODUCTION CELERY TASKS === #

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def process_pdf_task(self, files, metadata=None):
    """
    Main Celery task to:
        1) upload PDFs to S3, 
        2) save to Supabase, 
        3) and trigger vector embedding tasks.

    Args:
        files (List): Containing file CDN urls (created by WeWeb's document upload element)
        metadata (json): 

    Returns: 
        source_ids (List): list of source ids (for later use in the celery task chord)
        @TODO project_id: Associated project that is used
    """

    # === RENDER RESOURCE LOGGING (MB) === #
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    logger.info(f"Starting {self.name} with files: {files} and metadata: {metadata}")
    logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

    if not files:
        return {"error": "Please upload at least one PDF file before processing."}

    uploaded_documents = []
    source_ids = [] # used in chained task

    for file in files:
        try:
            # Handle URL or local file path
            if file.startswith('http://') or file.startswith('https://'):
                response = requests.get(file)
                response.raise_for_status()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_file.write(response.content)
                temp_file.close()
                file_path = temp_file.name
            else:
                file_path = file

            # Generate S3 key and upload file
            ext_ending = os.path.splitext(file_path)[1]
            s3_document_key = f"{uuid.uuid4()}{ext_ending}"
            upload_to_s3(s3_client, file_path, s3_document_key)
            
            # Generate CloudFront URL 
            cloudfront_document_url = get_cloudfront_url(s3_document_key)

            # Insert the document source record into Supabase (returns document_sources.id)
            source_id = insert_document_supabase_record(
                client=supabase_client,
                table_name="document_sources",
                cdn_url=cloudfront_document_url,
                content_tags=["Fitness", "Health", "Wearables"],   # @TODO: Lets make this AI taggedeventually
                uploaded_by=metadata.get("uploaded_by", "")
            )

            # Append to source_ids list
            source_ids.append(source_id)

            uploaded_documents.append({
                "source_id": source_id,
                "pdf_url": cloudfront_document_url,
                "file_path": file_path
            })

        except Exception as e:
            logger.error(f"Failed to upload and record file {file}: {e}", exc_info=True)
        finally:
            # Clean up temporary files if necessary
            if file.startswith('http'):
                os.unlink(file_path)

    # Trigger the embedding task for each document (chained Celery task)
    for doc in uploaded_documents:
        chunk_and_embed_task.delay(doc["pdf_url"], doc["source_id"])      # enqueue to Celery task queue
    
    # Logging message
    logger.info(" 'message' : PDF upload and record creation completed. Embedding tasks started.")

    return source_ids 

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def insert_project_documents_task(self, task_results):
    """
    Celery task to insert a row in the join TABLE `project_documents`. Project ⇆ Documents join associates each 
    project with N number of document_sources.

    Callback task to accept the list of results from the group tasks. The results will be in the 
    order the tasks were defined in the group.
    
    Args:
        task_results: (list) containing results from process_pdf_task and validate_and_generate_audio_task

    Returns: 
        None
    """
    # Unpack the results
    source_ids = task_results[0]
    media_result = task_results[1]
    project_id = media_result.get('media_id')

    try:
        for source_id in source_ids:
            supabase_client.table('project_documents').insert({
                'source_id': source_id,
                'media_id': project_id
            }).execute()
        logger.info(f"Successfully linked source_ids {source_ids} with media_id {project_id}")
    except Exception as e:
        logger.error(f"Failed to insert into TABLE `project_documents`: {e}")
        raise


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def insert_sources_media_association_task(self, task_results):
    """
    Celery task to insert a row in TABLE `sources_media_association` table. Callback task to accept 
    the list of results from the group tasks. The results will be in the order the tasks were defined in the group.
    
    - task_results: (list) containing results from process_pdf_task and validate_and_generate_audio_task
    """
    # Unpack the results
    source_ids = task_results[0]
    media_result = task_results[1]
    media_id = media_result.get('media_id')

    try:
        for source_id in source_ids:
            supabase_client.table('sources_media_association').insert({
                'source_id': source_id,
                'media_id': media_id
            }).execute()
        logger.info(f"Successfully linked source_ids {source_ids} with media_id {media_id}")
    except Exception as e:
        logger.error(f"Failed to insert into sources_media_association: {e}")
        raise

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def chunk_and_embed_task(self, pdf_url, source_id, chunk_size=1000, chunk_overlap=100):
    """
    Celery task to download, chunk, embed, and store embeddings of a PDF in Supabase. This step involved reading 
    the PDF contents with PyPDFLoader, but will need to be expanded to handle (.doc, .txt, .pdf (OCR), and .epub)
    """
    try:
        # Download the PDF file from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()
        
        # Load and split the PDF into chunks
        loader = PyPDFLoader(temp_file.name)        # performs OCR/extraction
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # Initialize embedding model
        embedding_model = OpenAIEmbeddings()

        # Process each chunk
        for chunk in chunks:
            try:
                chunk_text = chunk.page_content
                embedding = embedding_model.embed_query(chunk_text)
                
                # Insert the chunk and embedding into the Supabase 'document_vector_store' table
                insert_vector_supabase_record(
                    client=supabase_client,
                    table_name="document_vector_store",
                    source_id=source_id,
                    content=chunk_text,
                    metadata={
                        "source": pdf_url,
                        "page_number": chunk.metadata.get("page", None)
                    },
                    embedding=embedding
                )
            except Exception as e:
                logger.error(f"Error processing chunk: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to download and process file {pdf_url}: {e}", exc_info=True)
    finally:
        # Clean up temporary file
        os.unlink(temp_file.name)

    return {"message": f"Embedding task for {pdf_url} completed successfully."}