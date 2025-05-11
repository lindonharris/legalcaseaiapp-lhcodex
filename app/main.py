'''
Main FastAPI script, this is the heart of the web service
'''

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Dict, Any
import uuid
from utils.supabase_utils import supabase_client
from utils.pdf_utils import extract_text_from_pdf
from tasks.podcast_generate_tasks import validate_and_generate_audio_task, generate_dialogue_only_task
from tasks.upload_tasks import process_pdf_task, insert_sources_media_association_task
from tasks.test_tasks import addition_task
from tasks.chat_streaming_tasks import rag_chat_streaming_task
from tasks.chat_tasks import rag_chat_task
from tasks.note_tasks import rag_note_task 
from celery import chain, chord, group 
from celery.result import AsyncResult
import redis.asyncio as aioredis
import logging

# Configure logging (basic example, adjust as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Init FastAPI app & Redis pub.sub
app = FastAPI()
REDIS_LABS_URL = os.getenv("REDIS_LABS_URL_AND_PASS")  # REDIS_LABS_URL
redis_pub = aioredis.from_url(REDIS_LABS_URL, decode_responses=True)
# celery_app = Celery('tasks', broker='redis://localhost:6379/0')

origins = [
    "https://app.weweb.io",  # Replace with the actual WeWeb domain if different
    "https://editor.weweb.io",
    # Add other domains as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== PYDANTIC MODELS ======== #

class Numbers(BaseModel):
    x: int
    y: int
    
# Define Pydantic models for the responses
class SumResponse(BaseModel):
    sum: int

class PDFCaptureResponse(BaseModel):
    uuid: str
    url: str

# Define Pydantic model for the PDF extraction response
class PDFExtractResponse(BaseModel):
    filename: str
    combined_text: str

class PDFExtractBatchResponse(BaseModel):
    results: List[PDFExtractResponse]

# [SANITY CHECK] Request model for addition sanity
class AdditionRequest(BaseModel):
    x: int
    y: int

# Pydantic model for the request body
class PDFRequest(BaseModel):
    ''' WeWeb specific pydantic struct '''
    files: List[str]  # List of URLs or file paths of the PDFs
    metadata: Dict[str, Any]  # A dictionary for any metadata information

# New RAG pipeline request model
class NewRagPipelineRequest(BaseModel):
    ''' WeWeb specific pydantic struct for creation of a new RAG project '''
    files: List[str]  # List of URLs or file paths of the PDFs
    metadata: Dict[str, Any]  # A dictionary for any metadata information

# New RAG pipeline response model
class NewRagPipelineResponse(BaseModel):
    embedding_task_id: str      # from vector embedding task

# Chained RAG pipeline response model
class NewRagAndNoteResponse(BaseModel):
    workflow_id: str             # the chain’s overall task ID

class SourceIdsRequest(BaseModel):
    """Request body model for the status endpoint for checking the RAG pipeline upload."""
    source_ids: List[str]

# Pydantic model for the response
class PDFResponse(BaseModel):
    audio_task_id: str
    embedding_task_id: str

# RAG Query pydantic model
class RagQueryRequest(BaseModel):
    user_id: str
    chat_session_id: str
    query: str
    # document_ids: List[str]
    project_id: str
    model_type: str


# ================================================ #
#                  WEBSOCKETS
# ================================================ #

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(session_id: str, websocket: WebSocket):
    await websocket.accept()
    pubsub = redis_pub.pubsub()
    await pubsub.subscribe(f"chat_result:{session_id}")

    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await websocket.send_text(message['data'])
    except WebSocketDisconnect:
        await pubsub.unsubscribe(f"chat_result:{session_id}")


# ================================================ #
#               TEST ENDPOINTS
# ================================================ #

# Root endpoint
@app.get("/")
async def root():
    return {"success": "Hello Server LegalNoteAI FastAPI App"}

# Endpoint for sanity check 
@app.post("/sum_two_num/")
async def sum_two_num(numbers: AdditionRequest):
    print("execute 2 sum")
    task = addition_task.delay(numbers.x, numbers.y)
    print(f"Task submitted: {task.id}")
    return {"task_id": task.id}

# Endpoint for capturing PDF info (sanity check endpoint)
@app.get("/pdf_capture", response_model=PDFCaptureResponse)
async def pdf_capture(file: str):
    unique_id = str(uuid.uuid4())  # Generate a unique UUID
    return {"uuid": unique_id, "url": file}

# Endpoint for extracting a single PDF's content
@app.get("/pdf_extract", response_model=PDFExtractResponse)
async def pdf_extract(file: str):
    try:
        # Extract text using the utility function
        combined_text = extract_text_from_pdf(file)
        return {"filename": file.split("/")[-1], "combined_text": combined_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint for extracting content from multiple PDFs
@app.post("/pdf_extract_batch", response_model=PDFExtractBatchResponse)
async def pdf_extract_batch(files: List[str]):
    results = []
    
    for file_url in files:
        try:
            # Extract text using the utility function
            combined_text = extract_text_from_pdf(file_url)
            filename = file_url.split("/")[-1]
            results.append(PDFExtractResponse(filename=filename, combined_text=combined_text))
        
        except Exception as e:
            # Append an error message for the current file
            results.append(PDFExtractResponse(filename=file_url, combined_text=f"Error: {str(e)}"))
    
    return {"results": results}

# POST endpoint for addition using Celery
@app.post("/celery_test_addition/")
async def celery_test_addition(request: AdditionRequest):
    try:
        # Enqueue the Celery task
        print(f"API DEBUG: {[request.x, request.y]}")
        task = addition_task.apply_async(args=[request.x, request.y])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================ #
#               RAG API ENDPOINTS
# ================================================ #
@app.post("/new-rag-project/", response_model=NewRagPipelineResponse)
async def create_new_rag_project(
    request: NewRagPipelineRequest, 
    background_tasks: BackgroundTasks
):
    '''
    Endpoint to ONLY create a RAG pipeline for user project:
        - process_pdf_task:
            input: request.files, request.metadata
            returns: result??
        
    Request contains:
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        job = process_pdf_task.apply_async(
            args=[request.files, request.metadata]
        )
        return {"embedding_task_id": job.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/new-rag-and-notes/", response_model=NewRagAndNoteResponse)
async def create_new_rag_project_and_gen_notes(
    request: NewRagPipelineRequest, 
    background_tasks: BackgroundTasks
):
    '''
    Endpoint to create BOTH a RAG pipeline and a "quick action" AI note, chaining 2 celery notes together
        - process_pdf_task():
            input: request.files, request.metadata
            returns: None

        - rag_note_task():
            input: request.metadata
            returns: None
        
    Request contains:
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        # 1) Kick off a chain: upload → note
        workflow = chain(
            process_pdf_task.s(request.files, request.metadata),
            rag_note_task.s(
                request.metadata["user_id"],
                request.metadata["chat_session_id"],
                request.metadata["summary_query"],  # e.g. your custom "Based on the docs…" prompt
                request.metadata["project_id"],
                request.metadata["model_type"]
            )
        )
        result: AsyncResult = workflow.apply_async()
        return {"workflow_id": result.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/new-rag-project-deprecated/", response_model=NewRagPipelineResponse)
# async def create_new_rag_project_DEPR(request: NewRagPipelineRequest, background_tasks: BackgroundTasks):
#     '''
#     @INVESTIGATE: Should i use this to kill 2 birds with one stone? Using chords
#     Endpoint to create both a RAG pipeline for AI notes in one call:
#         - process_pdf_task:
#             input: request.files, request.metadata
#             returns: source_ids
#         - process_pdf_task:
#             input: source_ids
#             returns: None
        
#     Request contains
#     project_id
#         request.files (List): list of pdf file links
#         request.metadata (json): {
#             project_id:
#         }
#     '''
#     try:

#         # Create signatures for the tasks
#         process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)        # Upload PDFs to AWS S3, and Supabase
#         # validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

#         # Create the group
#         task_group = group(
#             process_pdf_task_signature,
#             # validate_and_generate_audio_task_signature
#         )

#         # Create the chord with the group and callback
#         task_chord = chord(task_group)(insert_project_documents_task.s())

#         # The result of the chord is the AsyncResult of the callback task
#         chord_result = task_chord

#         # Get task IDs
#         process_pdf_task_id = chord_result.parent.results[0].id
#         # validate_and_generate_audio_task_id = chord_result.parent.results[1].id
#         insert_task_id = chord_result.id

#         # Return the task IDs to the client
#         return {
#             # "audio_task_id": validate_and_generate_audio_task_id,
#             "embedding_task_id": process_pdf_task_id,
#             "insert_task_id": insert_task_id,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@app.post("/add-to-rag-project/", response_model=NewRagPipelineResponse)
async def append_sources_to_project(request: NewRagPipelineRequest, background_tasks: BackgroundTasks):
    '''
    @TODO HAVE NOT STARTED !!!
    Endpoint to add new sources to an existing RAG pipeline for AI notes in one call:
        - process_pdf_task:
            input: request.files, request.metadata
            returns: source_ids
        
    Request contains
    project_id
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)        # Upload PDFs to AWS S3, and Supabase
        # validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            # validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_project_documents_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        # validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            # "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-rag-note/")
async def generate_rag_note(request: RagQueryRequest):
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_type
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat/")
async def rag_chat(request: RagQueryRequest):
    """Endpoint for the rag query responses (token streaming not enabled)"""
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_type
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat-stream/")
async def rag_chat_stream(request: RagQueryRequest):
    """Endpoint for the rag query responses (token streaming not ENABLED), still a work in progress"""
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_streaming_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_type
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-upload-status/{task_id}")
async def get_rag_upload_status(
    task_id: str, # Still useful to pass the initial task ID for context
    request_body: SourceIdsRequest # Accept source_ids in the request body
):
    """
    Check the status of the RAG document processing workflow by querying
    the vector_embed_status in the database for the provided source_ids.
    """
    source_ids = request_body.source_ids

    if not source_ids:
        return {
            "task_id": task_id, # Include initial task ID for reference
            "status": "INVALID_REQUEST",
            "message": "No source_ids provided in the request body."
        }

    try:
        # Query the database for the status of the relevant documents
        logger.debug(f"Querying DB for status of source_ids: {source_ids}")
        db_response = supabase_client.table("document_sources") \
                                    .select("id, vector_embed_status") \
                                    .in_("id", source_ids) \
                                    .execute()

        # Check for database query errors
        # if db_response._raw_error:
        #     logger.error(f"Database query failed for status check: {db_response._raw_error}", exc_info=True)
        #     # You might want to return a specific error status indicating DB issue
        #     raise HTTPException(status_code=500, detail=f"Database query failed: {db_response._raw_error}")

        documents = db_response.data

        if not documents:
            # This could happen if the provided source_ids don't exist
            return {
                "task_id": task_id, # Include initial task ID for reference
                "status": "DOCUMENTS_NOT_FOUND",
                "message": f"Could not find documents with IDs {source_ids} in the database."
            }

        # Determine overall status based on individual document statuses
        statuses = [doc.get("vector_embed_status") for doc in documents if doc.get("vector_embed_status")]

        if "FAILED" in statuses:
            overall_status = "WORKFLOW_FAILED"
            message = "One or more documents failed during embedding."
        elif all(status == "COMPLETE" for status in statuses):
            overall_status = "WORKFLOW_COMPLETE"
            message = "All documents processed and embedded successfully."
        # If not failed and not all complete, it's still in progress
        elif any(status in ["PENDING", "UPLOADING", "EMBEDDING"] for status in statuses):
            overall_status = "WORKFLOW_IN_PROGRESS"
            message = "Document embedding is in progress."
        else:
            # Fallback for unexpected statuses
            overall_status = "WORKFLOW_UNKNOWN_STATUS"
            message = "Could not determine overall workflow status from document statuses."
            logger.warning(f"Unexpected document statuses encountered for source_ids {source_ids}: {statuses}")


        # Prepare individual document statuses for the response
        document_statuses = [{"id": doc.get("id"), "status": doc.get("vector_embed_status")} for doc in documents]

        return {
            "task_id": task_id, # Include initial task ID for reference
            "status": overall_status, # This is the derived workflow status
            "message": message,
            "document_statuses": document_statuses # Detailed status per document
        }

    except Exception as e:
        logger.error(f"Error retrieving or processing database status for task {task_id} and source_ids {source_ids}: {e}", exc_info=True)
        # Return an error indicating the status check itself failed
        raise HTTPException(status_code=500, detail=f"Failed to retrieve workflow status: {e}")

@app.get("/rag-chat-status/{task_id}")
async def get_rag_chat_status(task_id: str):
    """
    Check task status and result of the rag_chat_task.run() task.
    Includes timeout if stuck in PENDING > 20s    
    """
    # A proxy object that allows you to fetch the status and result of any Celery task
    result = AsyncResult(task_id)

    # Retrieve task metadata
    task_meta = result.info if isinstance(result.info, dict) else {}
    now = datetime.now(timezone.utc)

    # Use explicitly set start_time if available
    start_time_str = task_meta.get("start_time")
    if start_time_str:
        try:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed = (now - start_time).total_seconds()
        except Exception:
            elapsed = None
    else:
        elapsed = None  # fallback if broker doesn't store this

    if result.state == "PENDING":
        if elapsed and elapsed > 20:
            return {
                "task_id": task_id,
                "status": "TIMEOUT",
                "elapsed_time": elapsed,
                "message": "RAG task has exceeded 20 seconds. You may retry."
            }
        return {"task_id": task_id, "status": "PENDING", "elapsed_time": elapsed}
    elif result.state == "SUCCESS":
        return {"task_id": task_id, "status": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {"task_id": task_id, "status": "FAILURE", "error": str(result.result)}
    else:
        return {"task_id": task_id, "status": result.state}

# ================================================ #
#              PODCAST API ENDPOINTS
# ================================================ #

@app.post("/pdf-to-dialogue/", response_model=PDFResponse)
async def pdf_to_dialogue(request: PDFRequest, background_tasks: BackgroundTasks):
    '''
    Endpoint to create both a pdf/RAG pipeline and create an AI podcast in one call:
        - process_pdf_task 
        - validate_and_generate_audio_task
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)
        validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_sources_media_association_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Endpoint to process PDFs and generate dialogue transcript
@app.post("/pdf-to-dialogue-transcript/", response_model=PDFResponse)
async def pdf_to_dialogue_transcript(request: PDFRequest, background_tasks: BackgroundTasks):
    try:
        # Enqueue the Celery task for dialogue generation
        print(f"API DEBUG: {request.files}")
        task = generate_dialogue_only_task.apply_async(
            args=[request.files]
        )
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Combined endpoint to check the status of any Celery task
@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    Purpose:
        A generic Celery task monitor. It can check the status of any Celery task, including:
        - rag_chat_task
        - pdf_to_dialogue_task
        - addition_task
        ...or any other task your system may launch
        No custom behavior or message formatting per task type.

    Limitations:
        No custom behavior or message formatting per task type.
   """
    try:
        # A proxy object that allows you to fetch the status and result of any Celery task
        task_result = AsyncResult(task_id)

        # Retrieve task metadata
        task_meta = task_result.info if isinstance(task_result.info, dict) else {}
        start_time_str = task_meta.get('start_time')
        elapsed_time = None

        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        if task_result.state == 'PENDING':
            return {
                "task_id": task_id,
                "status": "PENDING",
                "elapsed_time": elapsed_time
            }
        elif task_result.state == 'SUCCESS':
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": task_result.result,
                "elapsed_time": elapsed_time
            }
        elif task_result.state == 'FAILURE':
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": str(task_result.result),
                "elapsed_time": elapsed_time
            }
        else:
            return {
                "task_id": task_id,
                "status": task_result.state,
                "elapsed_time": elapsed_time
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


    except Exception as e:
        # Handle any unexpected exceptions
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")

# ================================================ #
#              DEV-TOOLS ENDPOINTS
# ================================================ #

@app.get("/debug-task/{task_id}")
async def debug_task(task_id: str):
    """
    Dev-only endpoint to inspect any Celery task’s state, metadata, and result.
    WARNING: Don’t expose this in production without access control.
    """
    try:
        result = AsyncResult(task_id)
        task_meta = result.info if isinstance(result.info, dict) else {}

        return {
            "task_id": task_id,
            "state": result.state,
            "is_ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "result": str(result.result),  # may be a full dict or traceback
            "metadata": task_meta,
            "traceback": result.traceback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving task: {str(e)}")
