# This file contains relevant audio proocessing and AI audio api function calls

import os
import json
import io
import glob
import time
import logging
import tempfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as cf
from utils.llm_utils import get_mp3, conditional_llm
from pypdf import PdfReader  # Use one PdfReader, preferably `pypdf`
from pydantic import BaseModel, ValidationError
from tempfile import NamedTemporaryFile
from typing import List, Literal
from tenacity import retry, retry_if_exception_type
from dotenv import load_dotenv  # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]

def generate_audio(
    files: list,
    openai_api_key: str = None,
    text_model: str = "gpt-4o-mini",
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
    api_base: str = None,
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
    edited_transcript: str = None,
    user_feedback: str = None,
    original_text: str = None,
    debug = False,
) -> tuple:
    '''
    Generate audio from a list of PDF files using an LLM and text-to-speech models.


    Args:
        files (list): List of PDF files.
        openai_api_key (str, optional): OpenAI API key is from the evironment.
        text_model (str, optional): LLM text model to use.
        audio_model (str, optional): Audio model for text-to-speech conversion.
        speaker_1_voice (str, optional): Voice for speaker 1.
        speaker_2_voice (str, optional): Voice for speaker 2.
        original_text (str): is the extracted pdf (or OCR) data
        ... (other args)

    Returns:
        tuple: (temporary_file_path.name: (str), transcript: (List[Dict]), combined_text: (str))
    '''
    # Validate API Key
    if not os.getenv("OPENAI_API_KEY") and not openai_api_key:
        raise ValueError("OpenAI API key is required")

    combined_text = original_text or ""

    # Extract text from PDF files if original_text is not provided
    if not combined_text:
        for file in files:
            try:
                # Check if file is a URL and download it
                if file.startswith('http://') or file.startswith('https://'):
                    response = requests.get(file)
                    response.raise_for_status()  # Raise an error for bad responses
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.write(response.content)
                    temp_file.close()
                    file_path = temp_file.name
                    print('temp pdf file downloaded :)')
                else:
                    # Treat as a local file
                    file_path = file

                # Read the PDF content
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    # Use a fallback for None values when extracting text
                    text = "\n\n".join([page.extract_text() or '' for page in reader.pages])
                    combined_text += text + "\n\n"

                # Remove the temporary file if it was a downloaded URL
                if file.startswith('http://') or file.startswith('https://'):
                    Path(file_path).unlink()

            except Exception as e:
                logger.error(f"Error reading PDF file '{file}': {e}")
                return {"dialogue_text": None, "error": f"Error extracting text from PDF: {e}"}

    # Configure the LLM based on selected model and api_base
    @retry(retry=retry_if_exception_type(ValidationError))
    @conditional_llm(model=text_model, api_base=api_base, api_key=openai_api_key)
    def generate_dialogue(
        text: str, 
        intro_instructions: str, 
        text_instructions: str, 
        scratch_pad_instructions: str, 
        prelude_dialog: str,
        podcast_dialog_instructions: str,
        edited_transcript: str = None, 
        user_feedback: str = None, 
    ) -> Dialogue:
        """
        This function collects arguments and defines a structure for the prompt, which the @llm decorator (from promptic) 
        then uses to interact with the language model (LLM). However, `generate_dialogue()` doesn't actually execute any logic itself; 
        its main purpose is to serve as a template for the prompt that will be sent to the LLM via the @llm decorator.

        Example:
        --------

        {intro_instructions}
        
        Here is the original input text:
        
        <input_text>
        {text}
        </input_text>

        {text_instructions}
        
        <scratchpad>
        {scratch_pad_instructions}
        </scratchpad>
        
        {prelude_dialog}
        
        <podcast_dialogue>
        {podcast_dialog_instructions}
        </podcast_dialogue>
        {edited_transcript}{user_feedback}
        """

    instruction_improve = 'Based on the original text, please generate an improved version of the dialogue by incorporating the edits, comments and feedback.'

    # Ensure edited_transcript and user_feedback are not None before concatenating
    edited_transcript_processed = "\nPreviously generated edited transcript, with specific edits and comments that I want you to carefully address:\n" + \
        "<edited_transcript>\n" + (edited_transcript or '') + "</edited_transcript>" if edited_transcript else ""

    user_feedback_processed = "\nOverall user feedback:\n\n" + (user_feedback or '')

    if edited_transcript_processed.strip() != '' or user_feedback_processed.strip() != '':
        user_feedback_processed = "<requested_improvements>" + user_feedback_processed + "\n\n" + instruction_improve + "</requested_improvements>"
    
    if debug:
        logger.info(edited_transcript_processed)
        logger.info(user_feedback_processed)
    
    # Generate the dialogue using the LLM
    llm_output = generate_dialogue(
        combined_text,
        intro_instructions=intro_instructions,
        text_instructions=text_instructions,
        scratch_pad_instructions=scratch_pad_instructions,
        prelude_dialog=prelude_dialog,
        podcast_dialog_instructions=podcast_dialog_instructions,
        edited_transcript=edited_transcript_processed,
        user_feedback=user_feedback_processed
    )

    # Generate audio from the transcript
    audio = b""
    transcript = ""
    characters = 0

    # with cf.ThreadPoolExecutor() as executor:
    with ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            voice = speaker_1_voice if line.speaker == "speaker-1" else speaker_2_voice
            future = executor.submit(get_mp3, line.text, voice, audio_model, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    # Ensure _temp directory exists in the root of the project
    temp_dir = os.path.join(os.getcwd(), "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Create a file path for the audio in the _temp directory
    audio_file_path = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")

    # Write the audio to the file
    with open(audio_file_path, "wb") as audio_file:
        audio_file.write(audio)

    # Clean up old files in the _temp directory that are older than 24 hours
    for file in glob.glob(f"{temp_dir}/*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    return audio_file_path, transcript, combined_text


def generate_only_dialogue_text(
    files: list,
    openai_api_key: str = None,
    text_model: str = "gpt-4o-mini",
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
    api_base: str = None,
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
    edited_transcript: str = None,
    user_feedback: str = None,
    original_text: str = None,
    debug = False,
) -> tuple:
    '''
    Generate text ONLY from a list of PDF files using an LLM models.


    Args:
        files (list): List of PDF files.
        openai_api_key (str, optional): OpenAI API key is from the evironment.
        text_model (str, optional): LLM text model to use.
        audio_model (str, optional): Audio model for text-to-speech conversion.
        speaker_1_voice (str, optional): Voice for speaker 1.
        speaker_2_voice (str, optional): Voice for speaker 2.
        ... (other args)

    Returns:
        tuple: (temporary_file.name: (str), transcript: (List[Dict]), combined_text: (str))
    '''
    # Validate API Key
    if not os.getenv("OPENAI_API_KEY") and not openai_api_key:
        print(os.getenv("OPENAI_API_KEY"))
        raise ValueError("OpenAI API key is required!!")

    combined_text = original_text or ""

    # Extract text from PDF files if original_text is not provided
    # If there's no original text, extract it from the uploaded files or URLs
    if not combined_text:
        for file in files:
            try:
                # Check if file is a URL and download it
                if file.startswith('http://') or file.startswith('https://'):
                    response = requests.get(file)
                    response.raise_for_status()  # Raise an error for bad responses
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.write(response.content)
                    temp_file.close()
                    file_path = temp_file.name
                    print('temp pdf file downloaded :)')
                else:
                    # Treat as a local file
                    file_path = file

                # Read the PDF content
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    # Use a fallback for None values when extracting text
                    text = "\n\n".join([page.extract_text() or '' for page in reader.pages])
                    combined_text += text + "\n\n"

                # Remove the temporary file if it was a downloaded URL
                if file.startswith('http://') or file.startswith('https://'):
                    Path(file_path).unlink()

            except Exception as e:
                logger.error(f"Error reading PDF file '{file}': {e}")
                return {"dialogue_text": None, "error": f"Error extracting text from PDF: {e}"}

    # Configure the LLM based on selected model and api_base
    @retry(retry=retry_if_exception_type(ValidationError))
    @conditional_llm(model=text_model, api_base=api_base, api_key=openai_api_key)
    def generate_dialogue(
        text: str, 
        intro_instructions: str, 
        text_instructions: str, 
        scratch_pad_instructions: str, 
        prelude_dialog: str,
        podcast_dialog_instructions: str,
        edited_transcript: str = None, 
        user_feedback: str = None, 
    ) -> Dialogue:
        """
        This function collects arguments and defines a structure for the prompt, which the @llm decorator (from promptic) 
        then uses to interact with the language model (LLM). However, `generate_dialogue()` doesn't actually execute any logic itself; 
        its main purpose is to serve as a template for the prompt that will be sent to the LLM via the @llm decorator.

        Example:
        --------

        {intro_instructions}
        
        Here is the original input text:
        
        <input_text>
        {text}
        </input_text>

        {text_instructions}
        
        <scratchpad>
        {scratch_pad_instructions}
        </scratchpad>
        
        {prelude_dialog}
        
        <podcast_dialogue>
        {podcast_dialog_instructions}
        </podcast_dialogue>
        {edited_transcript}{user_feedback}
        """

    instruction_improve = 'Based on the original text, please generate an improved version of the dialogue by incorporating the edits, comments and feedback.'

    # Ensure edited_transcript and user_feedback are not None before concatenating
    edited_transcript_processed = "\nPreviously generated edited transcript, with specific edits and comments that I want you to carefully address:\n" + \
        "<edited_transcript>\n" + (edited_transcript or '') + "</edited_transcript>" if edited_transcript else ""

    user_feedback_processed = "\nOverall user feedback:\n\n" + (user_feedback or '')

    if edited_transcript_processed.strip() != '' or user_feedback_processed.strip() != '':
        user_feedback_processed = "<requested_improvements>" + user_feedback_processed + "\n\n" + instruction_improve + "</requested_improvements>"
    
    if debug:
        logger.info (edited_transcript_processed)
        logger.info (user_feedback_processed)
    
    # Generate the dialogue using the LLM
    llm_output = generate_dialogue(
        combined_text,
        intro_instructions=intro_instructions,
        text_instructions=text_instructions,
        scratch_pad_instructions=scratch_pad_instructions,
        prelude_dialog=prelude_dialog,
        podcast_dialog_instructions=podcast_dialog_instructions,
        edited_transcript=edited_transcript_processed,
        user_feedback=user_feedback_processed
    )

    # DEBUG: save the llm output to temp folder to see the full output
    temp_dir = os.path.join(os.getcwd(), "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Path to the output file in the _temp/ directory
    output_file_path = os.path.join(temp_dir, "llm_output.txt")

    # Dump the output to a text file in the _temp/ directory
    with open(output_file_path, "w") as file:
        json.dump(llm_output.model_dump(), file, indent=4)

    # Return the text dialogue as a string
    return llm_output.model_dump()  # Use model_dump instead of dict()