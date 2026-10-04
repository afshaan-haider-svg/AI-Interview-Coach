"""Background process manager to start and monitor the FastAPI backend on Streamlit Cloud."""

import atexit
import os
import subprocess
import sys
import time
from typing import Optional

import streamlit as st

from frontend.api_client import check_backend_health

# Absolute path to repository root directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bridge_secrets_to_env() -> None:
    """Safely bridges Streamlit secrets to os.environ for backend subprocess inheritance.

    Ensures that secrets configured in Streamlit Cloud (such as GEMINI_API_KEY)
    are available to the backend subprocess via environment variables.
    Strictly avoids printing, logging, or exposing sensitive values.
    """
    try:
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                secret_val = str(st.secrets["GEMINI_API_KEY"]).strip()
                if secret_val and not os.environ.get("GEMINI_API_KEY"):
                    os.environ["GEMINI_API_KEY"] = secret_val
            if "API_BASE_URL" in st.secrets:
                url_val = str(st.secrets["API_BASE_URL"]).strip()
                if url_val and not os.environ.get("API_BASE_URL"):
                    os.environ["API_BASE_URL"] = url_val
    except Exception:
        # In local development without secrets.toml, ignore missing secrets safely
        pass


class BackendProcessManager:
    """Manages the FastAPI backend subprocess lifecycle in containerized/cloud environments."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.process: Optional[subprocess.Popen] = None

    def is_alive(self) -> bool:
        """Returns True if the subprocess was started and is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def start(self) -> None:
        """Spawns the Uvicorn FastAPI backend subprocess if not already healthy or alive."""
        if check_backend_health():
            return
        if self.is_alive():
            return

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.repo_root}{os.pathsep}{existing_pp}" if existing_pp else self.repo_root
        )

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                env=env,
            )
            atexit.register(self.stop)
        except Exception:
            self.process = None

    def stop(self) -> None:
        """Gracefully terminates the backend subprocess on application shutdown."""
        if self.process and self.is_alive():
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass


@st.cache_resource(show_spinner=False)
def get_backend_manager(repo_root: str) -> BackendProcessManager:
    """Returns a singleton BackendProcessManager cached across Streamlit reruns and sessions."""
    return BackendProcessManager(repo_root)


def ensure_backend_running(repo_root: Optional[str] = None, timeout_seconds: int = 25) -> bool:
    """Ensures that the FastAPI backend is running and responding to health checks.

    1. Bridges secrets safely to environment variables.
    2. Checks if backend is already healthy (local development or previous run).
    3. If unhealthy, triggers background spawn via cached singleton manager.
    4. Bounded polling for readiness with user-facing status feedback.
    """
    root = repo_root or REPO_ROOT
    bridge_secrets_to_env()

    # If backend is already running, return immediately without spawning
    if check_backend_health():
        return True

    manager = get_backend_manager(root)
    if not manager.is_alive():
        manager.start()

    # Bounded polling for backend readiness
    start_time = time.time()
    with st.spinner("Starting AI Interview Coach backend services..."):
        while time.time() - start_time < timeout_seconds:
            if check_backend_health():
                return True
            if manager.process and manager.process.poll() is not None:
                # Subprocess exited unexpectedly (e.g. startup error)
                break
            time.sleep(0.5)

    return check_backend_health()
