"""Centralized HTTP API client for communicating with the FastAPI backend."""

import os
from typing import Any, Dict, List, Optional
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


class APIError(Exception):
    """Encapsulates a user-friendly API error with HTTP status and detail."""

    def __init__(self, status_code: int, user_message: str, detail: Optional[str] = None):
        super().__init__(user_message)
        self.status_code = status_code
        self.user_message = user_message
        self.detail = detail or user_message


def _map_http_error(status_code: int, raw_detail: str) -> str:
    """Translates raw HTTP status codes and details into actionable, friendly UI messages."""
    if status_code == 400:
        return f"Invalid request: {raw_detail}" if raw_detail else "Invalid request data. Please check your inputs."
    elif status_code == 404:
        return f"Resource not found: {raw_detail}" if raw_detail else "The requested resource was not found."
    elif status_code == 409:
        return f"Conflict: {raw_detail}" if raw_detail else "Operation cannot be completed in the current session state."
    elif status_code == 413:
        return "The uploaded resume file exceeds the maximum allowed limit of 5 MB."
    elif status_code == 422:
        return f"Validation error: {raw_detail}" if raw_detail else "Please check all required form fields."
    elif status_code == 429:
        return "The AI service is temporarily experiencing high demand. Please wait a moment and try again."
    elif status_code == 500:
        return "An internal server error occurred in the backend. Please try again."
    elif status_code == 502:
        return "The AI model returned an unexpected response. Please retry."
    elif status_code == 503:
        return "The service is temporarily unavailable. Please try again shortly."
    return f"Request failed with status {status_code}: {raw_detail}"


def _handle_response(resp: requests.Response) -> Any:
    """Parses JSON response or raises a structured APIError."""
    try:
        data = resp.json()
    except Exception:
        data = {}

    if not resp.ok:
        raw_detail = ""
        if isinstance(data, dict):
            raw_detail = str(data.get("detail", ""))
        elif isinstance(data, str):
            raw_detail = data
        user_msg = _map_http_error(resp.status_code, raw_detail)
        raise APIError(resp.status_code, user_msg, raw_detail)

    return data


def check_backend_health() -> bool:
    """Checks whether the FastAPI backend is running and healthy."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "healthy"
    except Exception:
        return False


def get_or_create_demo_user() -> Dict[str, Any]:
    """Retrieves or provisions the default local demo user."""
    try:
        r = requests.post(f"{API_BASE_URL}/users/demo", timeout=10)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(
            503,
            "Backend service is not running. Please start the FastAPI backend server and refresh this page.",
        )
    except requests.RequestException as e:
        raise APIError(500, f"Failed to connect to backend: {str(e)}")


def upload_resume(user_id: int, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Uploads a candidate resume PDF via FastAPI."""
    try:
        files = {"file": (filename, file_bytes, "application/pdf")}
        data = {"user_id": str(user_id)}
        r = requests.post(f"{API_BASE_URL}/resumes/upload", data=data, files=files, timeout=30)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Resume upload failed: {str(e)}")


def create_job_description(
    user_id: int,
    title: str,
    description: str,
    company: Optional[str] = None,
) -> Dict[str, Any]:
    """Submits job description text to the backend."""
    try:
        payload = {
            "user_id": user_id,
            "title": title.strip(),
            "description": description.strip(),
            "company": company.strip() if company else None,
        }
        r = requests.post(f"{API_BASE_URL}/job-descriptions", json=payload, timeout=20)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Job description submission failed: {str(e)}")


def start_interview(
    user_id: int,
    interview_type: str,
    difficulty: str,
    total_questions: int,
    resume_id: Optional[int] = None,
    job_description_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Initializes a new interview session and retrieves the first question."""
    try:
        payload = {
            "user_id": user_id,
            "interview_type": interview_type,
            "difficulty": difficulty,
            "total_questions": total_questions,
            "resume_id": resume_id,
            "job_description_id": job_description_id,
        }
        r = requests.post(f"{API_BASE_URL}/interview/sessions/start", json=payload, timeout=60)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Failed to start interview: {str(e)}")


def submit_answer(
    session_id: int,
    user_id: int,
    answer: str,
    answer_method: str = "text",
) -> Dict[str, Any]:
    """Submits candidate answer to evaluate and obtain the adaptive next question."""
    try:
        payload = {
            "user_id": user_id,
            "answer": answer.strip(),
            "answer_method": answer_method,
        }
        r = requests.post(f"{API_BASE_URL}/interview/sessions/{session_id}/answer", json=payload, timeout=60)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Answer evaluation request failed: {str(e)}")


def transcribe_audio(audio_bytes: bytes, filename: str = "recording.wav") -> Dict[str, Any]:
    """Uploads audio recording to local speech-to-text transcription service."""
    try:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        r = requests.post(f"{API_BASE_URL}/speech/transcribe", files=files, timeout=60)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Voice transcription request failed: {str(e)}")


def get_interview_session(session_id: int, user_id: int) -> Dict[str, Any]:
    """Retrieves current session state, pending question, and history."""
    try:
        r = requests.get(f"{API_BASE_URL}/interview/sessions/{session_id}?user_id={user_id}", timeout=20)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Failed to retrieve session: {str(e)}")


def list_interview_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all interview sessions for the specified candidate."""
    try:
        r = requests.get(f"{API_BASE_URL}/interview/sessions?user_id={user_id}", timeout=20)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Failed to retrieve interview history: {str(e)}")


def generate_report(session_id: int, user_id: int) -> Dict[str, Any]:
    """Generates the final comprehensive interview coaching report."""
    try:
        r = requests.post(f"{API_BASE_URL}/interview/sessions/{session_id}/report?user_id={user_id}", timeout=60)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Report generation failed: {str(e)}")


def get_report(session_id: int, user_id: int) -> Dict[str, Any]:
    """Retrieves the persisted final report for a completed session."""
    try:
        r = requests.get(f"{API_BASE_URL}/interview/sessions/{session_id}/report?user_id={user_id}", timeout=20)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError(503, "Backend service is not running.")
    except requests.RequestException as e:
        raise APIError(500, f"Failed to fetch final report: {str(e)}")
