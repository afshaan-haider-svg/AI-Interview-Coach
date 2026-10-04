"""API endpoints for local speech-to-text audio transcription."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.speech import TranscriptionResponse, transcribe_audio_file

router = APIRouter(prefix="/speech", tags=["Speech"])


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe spoken answer audio locally",
    description=(
        "Accepts recorded audio (WAV, MP3, WEBM, etc.), runs local faster-whisper inference on CPU, "
        "and returns transcript for candidate review and editing. Zero cloud speech APIs, zero cost, "
        "and zero audio persistence."
    ),
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Recorded audio file to transcribe"),
) -> TranscriptionResponse:
    """Transcribes uploaded audio locally using faster-whisper without external APIs."""
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the uploaded audio file.",
        ) from e

    filename = file.filename or "recording.wav"
    return transcribe_audio_file(contents=contents, original_filename=filename)
