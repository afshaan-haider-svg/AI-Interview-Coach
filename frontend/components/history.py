"""Interview history component for reviewing past practice sessions and loading reports."""

import streamlit as st
from frontend.api_client import (
    APIError,
    generate_report,
    get_interview_session,
    get_report,
    list_interview_sessions,
)
from frontend.state import INTERVIEW_TYPE_OPTIONS, reset_interview_state


def render_history_page():
    """Renders the past interview sessions list with report viewing and resume capabilities."""
    st.markdown("## 📋 Interview History")
    st.markdown("Review your past interview practice sessions, track progress over time, and revisit coaching reports.")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("No active user session found.")
        return

    try:
        with st.spinner("Loading interview history from backend..."):
            sessions = list_interview_sessions(user_id)
            st.session_state.history_sessions = sessions
    except APIError as e:
        st.error(f"Failed to load interview history: {e.user_message}")
        return

    if not sessions:
        st.info("No interview practice sessions found yet. Complete your first interview to see performance history!")
        if st.button("🚀 Start Your First Interview", type="primary"):
            st.session_state.current_page = "setup"
            st.rerun()
        return

    for s in sessions:
        s_id = s["session_id"]
        track_key = s["interview_type"]
        track_label = INTERVIEW_TYPE_OPTIONS.get(track_key, track_key.capitalize())
        diff = s.get("initial_difficulty", "medium").capitalize()
        status = s["status"]
        cand_name = s.get("candidate_name") or "Demo Candidate"
        readiness_badge = s.get("readiness_label")
        q_done = s.get("completed_questions", 0)
        q_total = s.get("total_questions", 5)
        score = s.get("overall_score")
        started = s.get("started_at", "")[:10]
        has_report = s.get("report_available", False)

        readiness_html = (
            f"""<span style="margin-left: 0.5rem; font-size: 0.8rem; background-color: #EEF2FF; color: #4338CA; padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 600;">{readiness_badge}</span>"""
            if readiness_badge
            else ""
        )

        import textwrap
        card_html = textwrap.dedent(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 0.5rem; padding: 1rem 1.25rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 700; color: #1E293B; font-size: 1.05rem;">{cand_name} &mdash; {track_label}</span>
                    <span style="margin-left: 0.5rem;" class="badge badge-{'easy' if diff=='Easy' else 'medium' if diff=='Medium' else 'hard'}">{diff}</span>
                    {readiness_html}
                    <span style="margin-left: 0.25rem; font-size: 0.8rem; color: #64748B;">• {started}</span>
                </div>
                <div>
                    <span style="font-weight: 700; font-size: 1.1rem; color: #4F46E5;">
                        {f'{score:.1f} / 10' if score is not None else 'In Progress'}
                    </span>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.4rem;">
                Session #{s_id} | Status: <strong>{status.capitalize()}</strong> | Questions Evaluated: <strong>{q_done} of {q_total}</strong>
            </div>
        </div>
        """).strip()

        with st.container():
            st.markdown(card_html, unsafe_allow_html=True)

            col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 3])

            if has_report:
                with col_btn1:
                    if st.button(f"📊 View Report #{s_id}", key=f"btn_view_{s_id}"):
                        try:
                            with st.spinner("Fetching report..."):
                                rep = get_report(s_id, user_id)
                                st.session_state.final_report = rep
                                st.session_state.current_page = "report"
                                st.rerun()
                        except APIError as e:
                            st.error(f"Failed to load report: {e.user_message}")

            elif status == "completed":
                with col_btn1:
                    if st.button(f"⚡ Generate Report #{s_id}", key=f"btn_gen_{s_id}"):
                        try:
                            with st.spinner("Generating performance report with Gemini..."):
                                rep = generate_report(s_id, user_id)
                                st.session_state.final_report = rep
                                st.session_state.current_page = "report"
                                st.rerun()
                        except APIError as e:
                            st.error(f"Failed to generate report: {e.user_message}")

            elif status == "active":
                with col_btn1:
                    if st.button(f"▶️ Resume Interview #{s_id}", key=f"btn_res_{s_id}"):
                        try:
                            with st.spinner("Reconstructing active interview session..."):
                                sess_data = get_interview_session(s_id, user_id)
                                st.session_state.session_id = s_id
                                st.session_state.interview_status = "active"
                                st.session_state.current_question = sess_data.get("current_question")
                                st.session_state.current_question_number = sess_data.get("completed_question_count", 0) + 1
                                st.session_state.selected_question_count = sess_data.get("total_questions", 5)
                                st.session_state.selected_interview_type = sess_data.get("interview_type", "python")
                                st.session_state.show_feedback_view = False
                                st.session_state.current_page = "interview"
                                st.rerun()
                        except APIError as e:
                            st.error(f"Failed to resume session: {e.user_message}")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("➕ Start New Interview", type="primary"):
        reset_interview_state()
        st.rerun()
