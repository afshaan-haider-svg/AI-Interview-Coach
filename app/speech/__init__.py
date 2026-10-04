"""Speech recognition module for AI-Powered Interview Coach."""

from app.speech.schemas import TranscriptionResponse
from app.speech.service import get_whisper_model, transcribe_audio_file

__all__ = ["TranscriptionResponse", "get_whisper_model", "transcribe_audio_file"]
