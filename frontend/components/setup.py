"""Setup page component for configuring personalized interview sessions."""

import streamlit as st
from frontend.api_client import (
    APIError,
    create_job_description,
    start_interview,
    upload_resume,
)
from frontend.state import DIFFICULTY_OPTIONS, INTERVIEW_TYPE_OPTIONS


def render_setup_page():
    """Renders the interview configuration and optional document ingestion form."""
    st.markdown("## ⚙️ Interview Setup")
    st.markdown("Configure your interview practice track, starting difficulty, and optional context documents.")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("No active user session found. Please refresh or verify backend connectivity.")
        return

    # Clear previous notifications
    if st.session_state.get("error_message"):
        st.error(st.session_state.error_message)
        st.session_state.error_message = None

    if st.session_state.get("success_message"):
        st.success(st.session_state.success_message)
        st.session_state.success_message = None

    # Step 1: Optional Resume Upload
    with st.expander("📄 Step 1: Upload Resume (Optional)", expanded=True):
        st.caption("Upload your resume (PDF only, max 5 MB). Our RAG system will ground technical questions in your background.")
        uploaded_file = st.file_uploader(
            "Choose your Resume PDF",
            type=["pdf"],
            key="resume_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.session_state.get("resume_filename") != uploaded_file.name:
                if st.button("📤 Ingest Resume PDF", key="btn_upload_resume"):
                    try:
                        with st.spinner("Extracting and indexing resume..."):
                            res = upload_resume(user_id, uploaded_file.getvalue(), uploaded_file.name)
                            st.session_state.resume_id = res["id"]
                            st.session_state.resume_filename = uploaded_file.name
                            if res.get("candidate_name"):
                                st.session_state.candidate_name = res["candidate_name"]
                                st.success(f"✅ Welcome, **{res['candidate_name']}**! Resume indexed successfully.")
                            else:
                                st.success(f"✅ Resume '{uploaded_file.name}' indexed successfully!")
                    except APIError as e:
                        st.error(f"Upload failed: {e.user_message}")

        if st.session_state.get("resume_id"):
            cand = st.session_state.get("candidate_name")
            cand_info = f" | Candidate: **{cand}**" if cand and cand != "Demo Candidate" else ""
            st.info(f"Attached Resume: **{st.session_state.resume_filename}** (ID: {st.session_state.resume_id}){cand_info}")

    # Step 2: Optional Job Description
    with st.expander("💼 Step 2: Target Job Description (Optional)", expanded=True):
        st.caption("Paste the target job description to practice questions aligned with the role's specific requirements.")
        jd_title = st.text_input("Job Title", value="Software Engineer / AI Specialist", key="input_jd_title")
        jd_company = st.text_input("Company Name (Optional)", key="input_jd_company")
        jd_text = st.text_area("Job Description Requirements", height=120, key="input_jd_desc", placeholder="Paste job responsibilities, skills, and qualifications...")

        if jd_text.strip() and not st.session_state.get("job_description_id"):
            if st.button("📝 Attach Job Description", key="btn_save_jd"):
                try:
                    with st.spinner("Indexing target job description..."):
                        jd_res = create_job_description(user_id, jd_title, jd_text, jd_company)
                        st.session_state.job_description_id = jd_res["id"]
                        st.session_state.job_title = jd_title
                        st.success(f"✅ Job Description for '{jd_title}' attached!")
                except APIError as e:
                    st.error(f"Job description save failed: {e.user_message}")

        if st.session_state.get("job_description_id"):
            st.info(f"Attached Job Description: **{st.session_state.job_title}** (ID: {st.session_state.job_description_id})")

    # Step 3: Interview Track & Difficulty Configuration
    st.markdown("### 🎯 Step 3: Practice Track & Parameters")

    col_track, col_diff, col_count = st.columns([2, 1.5, 1.5])

    with col_track:
        track_keys = list(INTERVIEW_TYPE_OPTIONS.keys())
        track_labels = [INTERVIEW_TYPE_OPTIONS[k] for k in track_keys]
        selected_track_idx = track_keys.index(st.session_state.get("selected_interview_type", "python"))
        selected_label = st.selectbox("Interview Track", options=track_labels, index=selected_track_idx)
        st.session_state.selected_interview_type = track_keys[track_labels.index(selected_label)]

    with col_diff:
        diff_keys = list(DIFFICULTY_OPTIONS.keys())
        diff_labels = [DIFFICULTY_OPTIONS[k] for k in diff_keys]
        selected_diff_idx = diff_keys.index(st.session_state.get("selected_difficulty", "medium"))
        selected_diff_label = st.selectbox("Initial Difficulty", options=diff_labels, index=selected_diff_idx)
        st.session_state.selected_difficulty = diff_keys[diff_labels.index(selected_diff_label)]

    with col_count:
        st.session_state.selected_question_count = st.number_input(
            "Question Count",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("selected_question_count", 5)),
            step=1,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Start Interview CTA
    if st.button("🚀 Start Interview", type="primary", use_container_width=True):
        try:
            with st.spinner("Generating your personalized first question with Gemini..."):
                start_res = start_interview(
                    user_id=user_id,
                    interview_type=st.session_state.selected_interview_type,
                    difficulty=st.session_state.selected_difficulty,
                    total_questions=int(st.session_state.selected_question_count),
                    resume_id=st.session_state.get("resume_id"),
                    job_description_id=st.session_state.get("job_description_id"),
                )

                st.session_state.session_id = start_res["session_id"]
                st.session_state.interview_status = start_res["status"]
                st.session_state.current_question = start_res["current_question"]
                st.session_state.current_question_number = 1
                st.session_state.current_difficulty = start_res["current_question"]["difficulty"]
                st.session_state.show_feedback_view = False
                st.session_state.answer_text = ""
                st.session_state.current_page = "interview"
                st.rerun()

        except APIError as e:
            st.error(f"Failed to start interview session: {e.user_message}")
