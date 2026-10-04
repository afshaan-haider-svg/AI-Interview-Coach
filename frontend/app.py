import os
import sys

# Ensure repository root is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="AI-Powered Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.api_client import APIError, check_backend_health, get_or_create_demo_user
from frontend.components.history import render_history_page
from frontend.components.home import render_home_page
from frontend.components.interview import render_interview_page
from frontend.components.report import render_report_page
from frontend.components.setup import render_setup_page
from frontend.components.sidebar import render_sidebar
from frontend.state import init_session_state
from frontend.styles import apply_custom_styles


def main():
    """Main application loop managing routing, state initialization, and error boundaries."""
    # Apply modern CSS styles
    apply_custom_styles()

    # Initialize session state keys
    init_session_state()

    # Check backend connectivity
    is_healthy = check_backend_health()
    st.session_state.backend_connected = is_healthy

    # Ensure demo user exists
    if is_healthy and not st.session_state.get("user_id"):
        try:
            demo_user = get_or_create_demo_user()
            st.session_state.user_id = demo_user["id"]
            st.session_state.user_name = demo_user["name"]
            st.session_state.user_email = demo_user["email"]
        except APIError as e:
            st.warning(f"Demo user resolution warning: {e.user_message}")

    # Render persistent sidebar
    render_sidebar()

    # Show warning banner if backend is unreachable
    if not is_healthy:
        st.error(
            "⚠️ **Backend service is not running.**\n\n"
            "The Streamlit frontend communicates with the FastAPI backend at `http://127.0.0.1:8000`.\n\n"
            "Please start the backend server in a terminal:\n"
            "```bash\n"
            ".venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000\n"
            "```\n"
            "Then refresh this page."
        )
        return

    # Page Routing
    current_page = st.session_state.get("current_page", "home")

    if current_page == "home":
        render_home_page()
    elif current_page == "setup":
        render_setup_page()
    elif current_page == "interview":
        render_interview_page()
    elif current_page == "report":
        render_report_page()
    elif current_page == "history":
        render_history_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
