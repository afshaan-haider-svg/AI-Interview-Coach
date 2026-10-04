"""Sidebar component with user profile, navigation, system status, and reset button."""

import streamlit as st
from frontend.api_client import check_backend_health
from frontend.state import reset_interview_state


def render_sidebar():
    """Renders the persistent left sidebar with navigation, candidate profile, and status."""
    with st.sidebar:
        st.markdown("### 🎯 Interview Coach")
        st.caption("AI-Powered Adaptive Practice")

        st.markdown("---")

        # Candidate Profile Box
        user_name = st.session_state.get("user_name", "Demo Candidate")
        user_email = st.session_state.get("user_email", "demo@example.com")
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: #1E293B; font-size: 0.95rem;">👤 {user_name}</div>
                <div style="font-size: 0.75rem; color: #64748B;">{user_email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation Options
        pages = ["Home", "Setup Interview"]

        if st.session_state.get("session_id") and st.session_state.get("interview_status") == "active":
            pages.append("Active Interview")

        if st.session_state.get("final_report") is not None:
            pages.append("Final Report")

        pages.append("Interview History")

        page_key_map = {
            "Home": "home",
            "Setup Interview": "setup",
            "Active Interview": "interview",
            "Final Report": "report",
            "Interview History": "history",
        }

        # Current page index
        current_page_key = st.session_state.get("current_page", "home")
        inv_map = {v: k for k, v in page_key_map.items()}
        curr_label = inv_map.get(current_page_key, "Home")
        if curr_label not in pages:
            curr_label = "Home"

        selected_page = st.radio(
            "Navigation",
            options=pages,
            index=pages.index(curr_label),
            label_visibility="collapsed",
        )

        st.session_state.current_page = page_key_map[selected_page]

        st.markdown("---")

        # Quick Actions
        if st.button("➕ Start New Interview", use_container_width=True):
            reset_interview_state()
            st.rerun()

        st.markdown("---")

        # Backend Health Check
        is_healthy = check_backend_health()
        st.session_state.backend_connected = is_healthy

        if is_healthy:
            st.markdown(
                '<div style="font-size: 0.8rem; color: #16A34A; font-weight: 600;">🟢 Backend Online</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size: 0.8rem; color: #DC2626; font-weight: 600;">🔴 Backend Offline</div>',
                unsafe_allow_html=True,
            )
            st.caption("Start FastAPI at `http://127.0.0.1:8000`")

        st.markdown(
            '<div style="margin-top: 2rem; font-size: 0.7rem; color: #94A3B8; text-align: center;">'
            "AI-Powered Interview Coach v1.0<br/>Phase 11 Production Frontend"
            "</div>",
            unsafe_allow_html=True,
        )
