"""Isolated local speech-to-text service using faster-whisper.

Runs 100% locally on CPU with int8 quantization. Zero paid API dependencies.
Implements lazy-loading singleton for the Whisper model and ensures all temporary
audio files are immediately deleted for candidate privacy.
"""

import os
import tempfile
import threading
from typing import Set
from fastapi import HTTPException, status

from app.speech.schemas import TranscriptionResponse

MAX_AUDIO_SIZE_MB = 25
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".webm",
    ".flac",
    ".aac",
}
DEFAULT_MODEL_SIZE = "base.en"

_model_lock = threading.Lock()
_cached_model = None


def get_whisper_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Lazy-loads and caches the local faster-whisper model in a thread-safe manner."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    with _model_lock:
        if _cached_model is None:
            from faster_whisper import WhisperModel

            # Load with CPU-optimized int8 quantization for low latency and small memory footprint
            _cached_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
    return _cached_model


def reset_cached_model() -> None:
    """Helper for testing to reset cached model instance."""
    global _cached_model
    with _model_lock:
        _cached_model = None


def transcribe_audio_file(
    contents: bytes,
    original_filename: str = "recording.wav",
    model_override=None,
) -> TranscriptionResponse:
    """Validates, safely stores temporarily, and transcribes audio via local faster-whisper.

    Strictly removes temporary files in a `finally` block to protect candidate privacy.
    Zero external cloud or paid speech APIs are used ($0.00 cost).
    """
    # 1. Validate file is non-empty
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty. Please record an answer before transcribing.",
        )

    # 2. Validate file size
    if len(contents) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Audio file exceeds maximum allowed size of {MAX_AUDIO_SIZE_MB} MB.",
        )

    # 3. Validate audio format extension
    _, ext = os.path.splitext(original_filename.lower())
    if ext and ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}",
        )

    # 4. Create isolated temporary file
    safe_ext = ext if ext in SUPPORTED_AUDIO_EXTENSIONS else ".wav"
    temp_file = tempfile.NamedTemporaryFile(suffix=safe_ext, delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(contents)
        temp_file.flush()
        temp_file.close()

        # 5. Retrieve lazy-loaded local Whisper model
        if model_override is not None:
            model = model_override
        else:
            try:
                model = get_whisper_model()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize local speech recognition engine: {str(e)}",
                ) from e

        # 6. Execute local transcription
        try:
            segments, info = model.transcribe(
                temp_path,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            collected_segments = list(segments)
            transcript_text = " ".join([seg.text.strip() for seg in collected_segments]).strip()
            duration = getattr(info, "duration", None)
        except Exception as e:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=f"Unable to process or decode the audio recording: {str(e)}",
            ) from e

        # 7. Check if speech was detected
        if not transcript_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No clear speech was detected in the recording. Please speak clearly and try again.",
            )

        return TranscriptionResponse(
            transcript=transcript_text,
            language=getattr(info, "language", "en") or "en",
            duration_seconds=round(float(duration), 2) if duration else None,
        )

    finally:
        # 8. Strict candidate privacy guarantee: immediately remove temporary audio file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
