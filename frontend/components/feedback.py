"""Feedback component displaying 5-dimension scoring, adaptive decisions, and detailed turn evaluations."""

import streamlit as st
from frontend.api_client import APIError, generate_report
from frontend.state import format_adaptive_reason


def render_feedback_view():
    """Renders the detailed feedback card after a candidate answer has been evaluated."""
    eval_data = st.session_state.get("latest_evaluation")
    adaptive_data = st.session_state.get("latest_adaptive_decision")
    next_q = st.session_state.get("next_question_buffer")
    session_id = st.session_state.get("session_id")
    user_id = st.session_state.get("user_id")
    status = st.session_state.get("interview_status")

    if not eval_data:
        st.warning("No evaluation data to display.")
        st.session_state.show_feedback_view = False
        st.rerun()
        return

    st.markdown("## 📋 Turn Evaluation & Feedback")

    overall = eval_data.get("overall_score", 0.0)
    scores = eval_data.get("scores", {})

    # Top Metric Grid
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("Overall Score", f"{overall:.1f} / 10")
    with m_col2:
        st.metric("Relevance", f"{scores.get('relevance', 0)} / 10")
    with m_col3:
        st.metric("Clarity", f"{scores.get('clarity', 0)} / 10")
    with m_col4:
        st.metric("Completeness", f"{scores.get('completeness', 0)} / 10")
    with m_col5:
        st.metric("Technical", f"{scores.get('technical_correctness', 0)} / 10")
    with m_col6:
        st.metric("Structure", f"{scores.get('structure', 0)} / 10")

    # Progress bars for dimension scores
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    pb1, pb2 = st.columns(2)
    with pb1:
        st.caption(f"Relevance: {scores.get('relevance', 0)}/10")
        st.progress(scores.get('relevance', 0) / 10.0)
        st.caption(f"Clarity: {scores.get('clarity', 0)}/10")
        st.progress(scores.get('clarity', 0) / 10.0)
        st.caption(f"Completeness: {scores.get('completeness', 0)}/10")
        st.progress(scores.get('completeness', 0) / 10.0)
    with pb2:
        st.caption(f"Technical Correctness: {scores.get('technical_correctness', 0)}/10")
        st.progress(scores.get('technical_correctness', 0) / 10.0)
        st.caption(f"Answer Structure: {scores.get('structure', 0)}/10")
        st.progress(scores.get('structure', 0) / 10.0)

    # Adaptive Difficulty Shift Banner
    if adaptive_data:
        prev_diff = adaptive_data.get("previous_difficulty", "").capitalize()
        next_diff = adaptive_data.get("next_difficulty", "").capitalize()
        reason = format_adaptive_reason(adaptive_data.get("reason_code"))

        st.markdown(
            f"""
            <div class="adaptive-banner">
                <strong>🎯 Adaptive Difficulty Shift:</strong> {prev_diff} &rarr; <strong>{next_diff}</strong><br/>
                <span style="font-size: 0.9rem;">Strategy Reason: {reason}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Strengths and Improvements
    col_str, col_imp = st.columns(2)
    with col_str:
        st.markdown("### ✅ Key Strengths")
        for s in eval_data.get("strengths", []):
            st.markdown(f"- {s}")

    with col_imp:
        st.markdown("### ⚠️ Areas for Improvement")
        for imp in eval_data.get("improvements", []):
            st.markdown(f"- {imp}")

    # Covered vs Missing Topics
    st.markdown("### 🏷️ Topic Alignment")
    cov = eval_data.get("covered_topics", [])
    mis = eval_data.get("missing_topics", [])

    if cov:
        st.markdown("**Covered Topics:** " + " ".join([f"<span class='topic-chip topic-covered'>✓ {t}</span>" for t in cov]), unsafe_allow_html=True)
    if mis:
        st.markdown("**Missing Topics:** " + " ".join([f"<span class='topic-chip topic-missing'>✗ {t}</span>" for t in mis]), unsafe_allow_html=True)

    # Technical Feedback
    tech_fb = eval_data.get("technical_feedback", [])
    if tech_fb:
        with st.expander("🔍 Detailed Technical Feedback", expanded=True):
            for fb in tech_fb:
                st.markdown(f"- {fb}")

    # Improved / Benchmark Answer
    sample = eval_data.get("improved_answer")
    if sample:
        with st.expander("💡 View Benchmark Sample Answer"):
            st.markdown(sample)

    st.markdown("---")

    # Action / Progression Buttons
    is_completed = (status == "completed") or (next_q is None)

    if is_completed:
        st.success("🎉 **Interview Complete!** All questions have been evaluated.")
        if st.button("📊 Generate Final Report", type="primary", use_container_width=True):
            try:
                with st.spinner("Synthesizing your comprehensive coaching analytics with Gemini..."):
                    report = generate_report(session_id, user_id)
                    st.session_state.final_report = report
                    st.session_state.current_page = "report"
                    st.session_state.show_feedback_view = False
                    st.rerun()
            except APIError as e:
                st.error(f"Report generation failed: {e.user_message}")
    else:
        if st.button("➡️ Continue to Next Question", type="primary", use_container_width=True):
            st.session_state.current_question = next_q
            st.session_state.current_question_number += 1
            st.session_state.current_difficulty = next_q.get("difficulty", "medium")
            st.session_state.show_feedback_view = False
            st.session_state.answer_text = ""
            st.session_state.voice_transcript = ""
            st.session_state.voice_audio_bytes = None
            st.session_state.transcription_error = None
            st.session_state.transcription_success = False
            st.session_state.is_submitting_answer = False
            st.session_state.next_question_buffer = None
            st.session_state.latest_evaluation = None
            st.session_state.latest_adaptive_decision = None
            st.rerun()
