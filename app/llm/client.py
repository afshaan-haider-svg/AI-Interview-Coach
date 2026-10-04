"""Gemini API client initialization and low-level content generation."""

import re
from typing import Optional
from fastapi import HTTPException, status
from google import genai
from google.genai import types

from app.config import settings


class GeminiLLMClient:
    """Reusable client for Google Gemini LLM API interactions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model if model is not None else settings.GEMINI_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self._client: Optional[genai.Client] = None

    def get_client(self) -> genai.Client:
        """Initializes and returns the Gemini client; validates API key existence."""
        if not self.api_key or not self.api_key.strip():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini API key is not configured. Please set GEMINI_API_KEY in the environment or .env file.",
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        """Executes content generation via the Gemini API with structured JSON request."""
        import time

        client = self.get_client()

        # Build candidate model sequence: primary model followed by verified fallback models
        candidate_models = [self.model]
        for fb in ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
            if fb not in candidate_models:
                candidate_models.append(fb)

        last_exception = None

        for model_name in candidate_models:
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            response_mime_type="application/json",
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        ),
                    )
                    if not response or not response.text:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Empty response received from the Gemini model.",
                        )
                    return response.text
                except HTTPException:
                    raise
                except Exception as e:
                    err_msg = str(e)
                    is_daily_quota = "quota" in err_msg.lower() or "limit: 20" in err_msg or "daily" in err_msg.lower()
                    if not is_daily_quota and ("503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_attempts - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    last_exception = e
                    # Try next candidate model if available
                    break

        # Map terminal error cleanly
        err_msg = str(last_exception) if last_exception else "Unknown LLM failure"
        if self.api_key:
            err_msg = err_msg.replace(self.api_key, "[REDACTED]")
        err_msg = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED]", err_msg)

        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Gemini API rate limit or quota exceeded: {err_msg}",
            ) from last_exception
        elif "503" in err_msg or "UNAVAILABLE" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Gemini API service temporarily unavailable: {err_msg}",
            ) from last_exception
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini API generation error: {err_msg}",
            ) from last_exception


_default_client: Optional[GeminiLLMClient] = None


def get_llm_client() -> GeminiLLMClient:
    """Singleton accessor for the default Gemini LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiLLMClient()
    return _default_client


def set_llm_client(client: Optional[GeminiLLMClient]) -> None:
    """Allows test suites to inject mock LLM clients."""
    global _default_client
    _default_client = client
