"""Interview turn component for displaying questions and accepting candidate answers."""

import streamlit as st
from frontend.api_client import APIError, submit_answer, transcribe_audio
from frontend.state import (
    INTERVIEW_TYPE_OPTIONS,
    validate_candidate_answer,
)


def render_interview_page():
    """Renders the active interview question and candidate answer input form."""
    session_id = st.session_state.get("session_id")
    curr_q = st.session_state.get("current_question")
    user_id = st.session_state.get("user_id")

    if session_id and not curr_q and user_id:
        try:
            from frontend.api_client import get_interview_session
            sess = get_interview_session(session_id, user_id)
            if sess.get("status") == "in_progress" and sess.get("current_question"):
                st.session_state.current_question = sess["current_question"]
                st.session_state.current_question_number = sess.get("current_question_number", 1)
                st.session_state.current_difficulty = sess.get("current_difficulty", "medium")
                st.session_state.interview_status = "in_progress"
                curr_q = st.session_state.current_question
            elif sess.get("status") == "completed":
                st.session_state.interview_status = "completed"
                st.session_state.current_page = "report"
                st.rerun()
        except Exception:
            pass

    if not session_id or not curr_q:
        st.warning("No active interview session in progress.")
        if st.button("Start Setup", type="primary"):
            st.session_state.current_page = "setup"
            st.rerun()
        return

    # Check if feedback view is active
    if st.session_state.get("show_feedback_view"):
        from frontend.components.feedback import render_feedback_view
        render_feedback_view()
        return

    total_q = st.session_state.get("selected_question_count", 5)
    curr_num = st.session_state.get("current_question_number", 1)
    track_key = st.session_state.get("selected_interview_type", "python")
    track_label = INTERVIEW_TYPE_OPTIONS.get(track_key, track_key.capitalize())
    diff = curr_q.get("difficulty", "medium").lower()
    category = curr_q.get("category", "technical")

    # Question Header & Metadata Badges
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 700; color: #4F46E5; font-size: 1.1rem;">
                Question {curr_num} of {total_q}
            </span>
            <div>
                <span class="badge badge-track">{track_label}</span>
                <span class="badge badge-{diff}">{diff.capitalize()}</span>
                <span class="badge badge-category">{category.capitalize()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Question Card
    q_text = curr_q.get("question", "")
    st.markdown(
        f"""
        <div class="question-card">
            <div style="font-size: 1.15rem; font-weight: 600; line-height: 1.6; color: #0F172A;">
                {q_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Answer Input Mode Selection
    answer_mode = st.radio(
        "Choose Answer Input Method:",
        options=["⌨️ Type Answer", "🎙️ Record Answer"],
        horizontal=True,
        key=f"ans_mode_radio_q{curr_num}",
    )

    if answer_mode == "⌨️ Type Answer":
        st.markdown("#### ✍️ Type Your Answer")
        answer_input = st.text_area(
            "Type your answer here",
            value=st.session_state.get("answer_text", ""),
            height=200,
            placeholder="Structure your answer clearly with explanations, examples, and technical trade-offs...",
            key=f"candidate_answer_input_q{curr_num}",
            label_visibility="collapsed",
        )
        final_answer = answer_input
        effective_method = "text"

    else:
        st.markdown("#### 🎙️ Record Your Answer")
        st.info("💡 Voice transcription currently works best with English answers.")

        audio_value = st.audio_input("Record audio answer", key=f"audio_input_q{curr_num}")

        if audio_value is not None:
            col_transcribe, _ = st.columns([2, 3])
            with col_transcribe:
                if st.button("⚡ Transcribe Answer", key=f"btn_transcribe_q{curr_num}", type="secondary", use_container_width=True):
                    try:
                        with st.spinner("Transcribing your spoken answer locally with Whisper..."):
                            res = transcribe_audio(
                                audio_bytes=audio_value.getvalue(),
                                filename=audio_value.name or "recording.wav",
                            )
                            transcript = (res.get("transcript") or "").strip()
                            if transcript:
                                st.session_state["voice_transcript"] = transcript
                                st.session_state["transcription_success"] = True
                                st.session_state["transcription_error"] = None
                                st.rerun()
                            else:
                                st.session_state["transcription_success"] = False
                                st.session_state["transcription_error"] = (
                                    "No clear speech was detected in your recording. Please speak clearly and try again."
                                )
                    except APIError as e:
                        st.session_state["transcription_success"] = False
                        st.session_state["transcription_error"] = (
                            "We couldn't transcribe this recording. Please try recording again or switch to a typed answer.\n\n"
                            f"Detail: {e.user_message}"
                        )

        if st.session_state.get("transcription_error"):
            st.error(st.session_state["transcription_error"])

        if (
            st.session_state.get("transcription_success")
            and st.session_state.get("voice_transcript")
            and st.session_state.get("voice_transcript", "").strip()
        ):
            st.success("✅ Speech transcribed successfully! Please review and edit below before submitting.")

        st.markdown("##### 📝 Review & Edit Transcript Before Submitting")
        st.caption("You can manually correct technical terms (e.g. LangGraph, RAG, ChromaDB, FastAPI) before evaluation.")

        if "voice_transcript" not in st.session_state:
            st.session_state["voice_transcript"] = ""

        edited_transcript = st.text_area(
            "Editable Transcript",
            key="voice_transcript",
            height=180,
            placeholder="Recorded speech transcript will appear here after clicking 'Transcribe Answer'. You can edit it manually before submitting...",
            label_visibility="collapsed",
        )
        final_answer = edited_transcript
        effective_method = "voice"

    col1, col2 = st.columns([1.5, 3])
    with col1:
        submit_clicked = st.button("🚀 Submit Answer", type="primary", use_container_width=True)

    if submit_clicked:
        if not validate_candidate_answer(final_answer):
            if answer_mode == "🎙️ Record Answer":
                st.error("Please record an answer and click '⚡ Transcribe Answer' first, or switch to '⌨️ Type Answer'.")
            else:
                st.error("Please enter a meaningful answer before submitting. Blank answers cannot be evaluated.")
            return

        clean_final = final_answer.strip()

        if st.session_state.get("is_submitting_answer", False):
            return

        st.session_state["is_submitting_answer"] = True
        try:
            with st.spinner("Analyzing and evaluating your answer with Gemini..."):
                ans_res = submit_answer(
                    session_id=session_id,
                    user_id=user_id,
                    answer=clean_final,
                    answer_method=effective_method,
                )

                st.session_state["latest_evaluation"] = ans_res.get("evaluation")
                st.session_state["latest_adaptive_decision"] = ans_res.get("adaptive_decision")
                st.session_state["next_question_buffer"] = ans_res.get("next_question")
                st.session_state["interview_status"] = ans_res.get("status")
                st.session_state["show_feedback_view"] = True
                st.session_state["answer_text"] = clean_final
                st.rerun()

        except APIError as e:
            st.error(f"Submission failed: {e.user_message}")
        finally:
            st.session_state["is_submitting_answer"] = False
