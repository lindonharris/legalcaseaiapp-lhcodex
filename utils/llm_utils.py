import os
import io
import concurrent.futures as cf
from pathlib import Path
import tempfile
import glob
import time
import logging

from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from typing import List, Literal


def get_mp3(
        text: str, 
        voice: str, 
        audio_model: str, 
        api_key: str = None) -> bytes:
    '''
    Sends text to the OpenAI Text-to-Speech (TTS) API and returns the generated MP3 audio file.

    Parameters:
        text (str) : The input text to be converted to speech.
        voice (str) : The desired voice for the TTS output. Must be a voice supported by the specified audio model.
        audio_model (str) : The audio model to use for generating the speech. This must be a valid model recognized by the API.
        api_key (str, optional) : The API key for authenticating with the OpenAI TTS service. If not provided, it assumes that the 
        environment is configured to authenticate without an explicit key.
    
    Returns:
        bytes: The MP3 audio file as a byte stream.
    
    Example:
    -------
    >>> audio_data = get_mp3("Hello, world!", "en-US-Standard-D", "standard")
    >>> with open('output.mp3', 'wb') as f:
    >>>     f.write(audio_data)
    '''
    
    client = OpenAI( api_key=api_key or os.getenv("OPENAI_API_KEY"),  )

    with client.audio.speech.with_streaming_response.create(
        model=audio_model,
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()
        

def conditional_llm(model, api_base=None, api_key=None):
    """
    Conditionally apply the @llm decorator based on the api_base parameter.
    If api_base is provided, it applies the @llm decorator with api_base.
    Otherwise, it applies the @llm decorator without api_base.

    In other words: llm(model=model, api_key=api_key)(func) is executed and returned...

    Args:
        model (str) : string describing the model name e.g. "o1-preview-2024-09-12"
        api_base:
        api_key: (.env) variable from os.getenv("OPENAI_API_KEY")
    """
    def decorator(func):
        if api_base:
            return llm(model=model, api_base=api_base)(func)
        else:
            return llm(model=model, api_key=api_key)(func)
    return decorator