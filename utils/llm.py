import base64
import logging
import os
import random
import time
from functools import wraps
from io import BytesIO
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel

from google import genai

logger = logging.getLogger(__name__)

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

T = TypeVar("T", bound=BaseModel)


def retry_with_backoff(
    max_retries=3, initial_delay=1, max_delay=10, exponential_base=2
):
    """
    Decorator that implements exponential backoff for rate-limited API calls.

    Args:
        max_retries (int): Maximum number of retry attempts
        initial_delay (int): Initial delay in seconds
        max_delay (int): Maximum delay in seconds
        exponential_base (int): Base for exponential backoff calculation
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    if "429" in str(e):
                        if retries == max_retries - 1:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded. Last error: {str(e)}"
                            )
                            raise

                        # Calculate delay with jitter
                        jitter = random.uniform(0, 0.1 * delay)
                        current_delay = min(delay + jitter, max_delay)

                        logger.warning(
                            f"Rate limit hit. Retrying in {current_delay:.1f} seconds (Attempt {retries + 1}/{max_retries})"
                        )
                        time.sleep(current_delay)

                        # Increase delay exponentially
                        delay = min(delay * exponential_base, max_delay)
                        retries += 1
                    else:
                        # If it's not a rate limit error, raise immediately
                        logger.error(f"Error in LLM call: {str(e)}")
                        raise
            return None

        return wrapper

    return decorator


def _prepare_file_for_upload(file_data: Union[str, bytes, Image.Image], mime_type: Optional[str] = None) -> tuple:
    """
    Prepare a file for upload to Gemini API.
    
    Args:
        file_data: The file as a path string, bytes, or PIL image
        mime_type: Optional mime type to use (otherwise determined automatically)
        
    Returns:
        Tuple of (file_bytes, mime_type)
    """
    if isinstance(file_data, str):
        # It's a file path
        with open(file_data, 'rb') as f:
            file_bytes = f.read()
            
        # Determine mime type if not provided
        if not mime_type:
            if file_data.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif file_data.lower().endswith('.png'):
                mime_type = 'image/png'
            elif file_data.lower().endswith('.mp3'):
                mime_type = 'audio/mp3'
            elif file_data.lower().endswith('.wav'):
                mime_type = 'audio/wav'
            elif file_data.lower().endswith('.ogg'):
                mime_type = 'audio/ogg'
            elif file_data.lower().endswith('.flac'):
                mime_type = 'audio/flac'
            elif file_data.lower().endswith('.aac'):
                mime_type = 'audio/aac'
            elif file_data.lower().endswith('.aiff'):
                mime_type = 'audio/aiff'
            else:
                raise ValueError(f"Could not determine mime type for {file_data}")
                
    elif isinstance(file_data, bytes):
        # It's raw bytes
        file_bytes = file_data
        if not mime_type:
            raise ValueError("mime_type must be provided when passing bytes")
    elif isinstance(file_data, Image.Image):
        # It's a PIL Image
        buffered = BytesIO()
        file_data.save(buffered, format="JPEG", quality=95)
        file_bytes = buffered.getvalue()
        mime_type = 'image/jpeg'
    else:
        raise ValueError(f"Unsupported file_data type: {type(file_data)}")
        
    return file_bytes, mime_type


@retry_with_backoff()
def call(
    prompt: str,
    image: Optional[Union[str, Image.Image, bytes]] = None,
    audio: Optional[Union[str, bytes]] = None,
    response_model: Optional[Type[T]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Calls the Gemini LLM with the given prompt and optional media, with retries.

    Args:
        prompt (str): The text prompt to send to the LLM.
        image (Optional[Union[str, Image.Image, bytes]]): Optional image as a PIL Image,
            file path, or bytes.
        audio (Optional[Union[str, bytes]]): Optional audio as a file path or bytes.
        response_model (Optional[Type[T]]): Optional Pydantic BaseModel for structured output.

    Returns:
        Union[str, Dict[str, Any]]: The LLM's response as a string, or a dictionary if a response_model
            is provided. Returns an empty string or an empty dictionary (as appropriate) on failure, after retries.
    """
    try:
        contents = [prompt]

        # Handle image if provided
        if image:
            image_bytes, image_mime = _prepare_file_for_upload(image)
            
            # Check size to determine upload method
            if len(image_bytes) > 20 * 1024 * 1024:  # 20MB
                # Large file - use File API
                with BytesIO(image_bytes) as f:
                    image_file = client.files.upload(file=f, mime_type=image_mime)
                contents.append(image_file)
            else:
                # Small file - use inline
                contents.append(
                    genai.types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=image_mime
                    )
                )

        # Handle audio if provided
        if audio:
            audio_bytes, audio_mime = _prepare_file_for_upload(audio)
            
            # Check size to determine upload method
            if len(audio_bytes) > 20 * 1024 * 1024:  # 20MB
                # Large file - use File API
                with BytesIO(audio_bytes) as f:
                    audio_file = client.files.upload(file=f, mime_type=audio_mime)
                contents.append(audio_file)
            else:
                # Small file - use inline
                contents.append(
                    genai.types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=audio_mime
                    )
                )

        # Generate content
        if response_model:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=contents,
                structured_output=response_model,
            )
            logger.info(f"LLM response: {response}")
            return response.structured_output
        else:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=contents,
            )
            logger.info(f"LLM response: {response.text}")
            return response.text
    
    except Exception as e:
        logger.error(f"Error during LLM call: {e}")
        if response_model:
            return {}
        else:
            return ""


def transcribe_audio(audio_file: Union[str, bytes]) -> str:
    """
    Transcribe an audio file using Gemini.
    
    Args:
        audio_file: Path to audio file or bytes
        
    Returns:
        Transcription as a string
    """
    return call("Generate a transcript of the speech, include diarization info - speakers, emotions in xml tags", audio=audio_file)
