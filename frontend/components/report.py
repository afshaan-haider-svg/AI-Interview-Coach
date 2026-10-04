"""Final report component providing performance dashboards and qualitative coaching analytics."""

import streamlit as st
from frontend.state import INTERVIEW_TYPE_OPTIONS, reset_interview_state


def render_report_page():
    """Renders the comprehensive performance report dashboard and per-turn breakdown."""
    report = st.session_state.get("final_report")

    if not report:
        st.warning("No interview report is currently loaded.")
        if st.button("Go to Setup", type="primary"):
            st.session_state.current_page = "setup"
            st.rerun()
        return

    track_key = report.get("interview_type", "python")
    track_label = INTERVIEW_TYPE_OPTIONS.get(track_key, track_key.capitalize())
    candidate_name = report.get("candidate_name") or st.session_state.get("candidate_name", "Demo Candidate")
    analytics = report.get("analytics", {})
    dim_avgs = analytics.get("dimension_averages", {})
    overall = analytics.get("overall_score", 0.0)
    q_count = analytics.get("question_count", len(report.get("questions", [])))

    # Header Banner
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%); padding: 1.75rem 2rem; border-radius: 0.75rem; color: white; margin-bottom: 1.5rem;">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Personalized Interview Report</div>
            <div style="font-size: 2rem; font-weight: 800; margin-top: 0.25rem;">{candidate_name} &mdash; {track_label} Interview Report</div>
            <div style="display: flex; gap: 2rem; margin-top: 1rem; align-items: baseline;">
                <div><span style="font-size: 2.2rem; font-weight: 800; color: #818CF8;">{overall:.2f}</span> <span style="opacity: 0.8;">/ 10 Overall Score</span></div>
                <div><span style="font-size: 1.2rem; font-weight: 600;">{q_count}</span> <span style="opacity: 0.8;">Questions Evaluated</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interview Readiness Assessment Card
    readiness = report.get("readiness") or {}
    readiness_label = readiness.get("readiness_label", "Developing — More Preparation Recommended")
    readiness_code = readiness.get("readiness_code", "developing")
    headline_wording = readiness.get("headline_wording", f"Interview readiness assessment for {track_label}.")
    assessment_message = readiness.get("assessment_message", "")
    priority_improvements = readiness.get("priority_improvement_areas", [])
    strongest_areas = readiness.get("strongest_areas", [])
    next_step = readiness.get("next_step", "")

    badge_bg = "#D1FAE5" if readiness_code == "strong_readiness" else "#DBEAFE" if readiness_code == "developing" else "#FFEDD5"
    badge_color = "#065F46" if readiness_code == "strong_readiness" else "#1E40AF" if readiness_code == "developing" else "#9A3412"

    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1.5px solid {badge_color}44; border-radius: 0.75rem; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #1E293B;">🎯 Interview Readiness Assessment</span>
                <span style="background-color: {badge_bg}; color: {badge_color}; font-weight: 700; padding: 0.35rem 0.85rem; border-radius: 9999px; font-size: 0.85rem;">
                    {readiness_label}
                </span>
            </div>
            <div style="font-size: 1rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;">
                {headline_wording}
            </div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem;">
                {assessment_message}
            </p>
            <div style="background-color: #F8FAFC; border-left: 4px solid {badge_color}; padding: 0.75rem 1rem; border-radius: 0.25rem; margin-bottom: 0.5rem;">
                <strong style="color: #1E293B;">📌 Actionable Next Step:</strong> <span style="color: #334155;">{next_step}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if priority_improvements or strongest_areas:
        col_rec_str, col_rec_imp = st.columns(2)
        with col_rec_str:
            st.markdown("#### 🌟 Top Demonstrated Strengths")
            for sa in strongest_areas:
                st.markdown(f"- ✅ **{sa}**")
        with col_rec_imp:
            st.markdown("#### 🎯 Priority Growth Opportunities")
            for pi in priority_improvements:
                st.markdown(f"- 🚀 {pi}")
        st.markdown("---")

    # Section A: Performance Overview
    st.markdown("### 📊 Performance Analytics Overview")
    col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
    with col_ov1:
        st.metric("Highest Question Score", f"{analytics.get('highest_question_score', 0):.1f} / 10")
    with col_ov2:
        st.metric("Lowest Question Score", f"{analytics.get('lowest_question_score', 0):.1f} / 10")
    with col_ov3:
        st.metric("Strongest Dimension", str(analytics.get("strongest_dimension", "N/A")).replace("_", " ").capitalize())
    with col_ov4:
        st.metric("Area for Growth", str(analytics.get("weakest_dimension", "N/A")).replace("_", " ").capitalize())

    # Section B: Dimension Scores
    st.markdown("### 🎯 Core Dimension Averages")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        rel = dim_avgs.get("relevance", 0.0)
        st.caption(f"Relevance: {rel:.2f} / 10")
        st.progress(min(1.0, rel / 10.0))

        cla = dim_avgs.get("clarity", 0.0)
        st.caption(f"Clarity: {cla:.2f} / 10")
        st.progress(min(1.0, cla / 10.0))

        comp = dim_avgs.get("completeness", 0.0)
        st.caption(f"Completeness: {comp:.2f} / 10")
        st.progress(min(1.0, comp / 10.0))

    with d_col2:
        tech = dim_avgs.get("technical_correctness", 0.0)
        st.caption(f"Technical Correctness: {tech:.2f} / 10")
        st.progress(min(1.0, tech / 10.0))

        stru = dim_avgs.get("structure", 0.0)
        st.caption(f"Structure & Organization: {stru:.2f} / 10")
        st.progress(min(1.0, stru / 10.0))

    st.markdown("---")

    # Section C: Executive Coaching Summary
    st.markdown("### 📝 Executive Coaching Summary")
    exec_summary = report.get("executive_summary", "")
    st.info(exec_summary)

    # Section D: Strengths vs Improvement Areas
    col_str, col_imp = st.columns(2)
    with col_str:
        st.markdown("#### ✅ Demonstrated Strengths")
        for s in report.get("key_strengths", []):
            st.markdown(f"- {s}")

    with col_imp:
        st.markdown("#### 🚀 Target Improvement Areas")
        for imp in report.get("improvement_areas", []):
            st.markdown(f"- {imp}")

    st.markdown("---")

    # Section E: Topic Gap Analysis & Recommendations
    col_top, col_tips = st.columns(2)
    with col_top:
        st.markdown("#### 📚 Targeted Study Recommendations")
        for rec in report.get("study_recommendations", []):
            st.markdown(f"- 📖 **{rec}**")

        topic_gaps = report.get("topic_gap_analysis", [])
        if topic_gaps:
            st.markdown("#### 🔍 Topic Gaps Identified")
            for gap in topic_gaps:
                st.markdown(f"- ⚠️ {gap}")

    with col_tips:
        st.markdown("#### 💡 Interview Delivery Tips")
        for tip in report.get("interview_coaching_tips", []):
            st.markdown(f"- 💡 {tip}")

        # Aggregated Covered vs Missing topics
        cov = analytics.get("covered_topics", [])
        mis = analytics.get("missing_topics", [])
        if cov or mis:
            st.markdown("#### 🏷️ Cumulative Topic Coverage")
            if cov:
                st.markdown("**Covered:** " + " ".join([f"<span class='topic-chip topic-covered'>✓ {t}</span>" for t in cov]), unsafe_allow_html=True)
            if mis:
                st.markdown("**Missing:** " + " ".join([f"<span class='topic-chip topic-missing'>✗ {t}</span>" for t in mis]), unsafe_allow_html=True)

    st.markdown("---")

    # Section F: Difficulty Progression
    progression = analytics.get("difficulty_progression", [])
    if progression:
        st.markdown("### 📈 Difficulty Trajectory")
        prog_str = " &rarr; ".join([f"<strong>{d.capitalize()}</strong>" for d in progression])
        changes = analytics.get("difficulty_changes", {}) or {}
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
                <div>{prog_str}</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.5rem;">
                    Increases: {changes.get('increases', 0)} | Maintained: {changes.get('maintains', 0)} | Decreases: {changes.get('decreases', 0)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section G: Question-by-Question Review
    st.markdown("### 🔍 Question-by-Question Detailed Review")
    questions = report.get("questions", [])

    for q in questions:
        q_num = q.get("question_number", 1)
        q_score = q.get("overall_score", 0.0)
        q_diff = q.get("difficulty", "medium").capitalize()
        with st.expander(f"Question {q_num} — {q_diff} — Score: {q_score:.1f} / 10", expanded=False):
            st.markdown(f"**Question:** {q.get('question')}")
            st.markdown(f"**Your Answer:** {q.get('candidate_answer')}")

            q_scores = q.get("scores", {})
            st.caption(
                f"Scores: Relevance: {q_scores.get('relevance')}/10 | Clarity: {q_scores.get('clarity')}/10 | "
                f"Completeness: {q_scores.get('completeness')}/10 | Technical: {q_scores.get('technical_correctness')}/10 | "
                f"Structure: {q_scores.get('structure')}/10"
            )

            col_q_str, col_q_imp = st.columns(2)
            with col_q_str:
                st.markdown("**Strengths:**")
                for s in q.get("strengths", []):
                    st.markdown(f"- {s}")
            with col_q_imp:
                st.markdown("**Areas to Improve:**")
                for imp in q.get("improvements", []):
                    st.markdown(f"- {imp}")

            if q.get("technical_feedback"):
                st.markdown("**Technical Notes:**")
                for tf in q.get("technical_feedback", []):
                    st.markdown(f"- {tf}")

            if q.get("improved_answer"):
                st.markdown(f"**Benchmark Improved Answer:**\n\n{q.get('improved_answer')}")

    st.markdown("---")

    # Bottom Actions
    # Build exportable text/markdown
    export_lines = [
        f"# {candidate_name} — {track_label} Interview Report",
        "",
        f"- **Candidate:** {candidate_name}",
        f"- **Track:** {track_label}",
        f"- **Overall Score:** {overall:.2f} / 10",
        f"- **Interview Readiness:** {readiness_label}",
        f"- **Questions Evaluated:** {q_count}",
        "",
        "## Executive Coaching Summary",
        f"{report.get('executive_summary', '')}",
        "",
        "## Interview Readiness Assessment",
        f"**{headline_wording}**",
        "",
        f"{assessment_message}",
        "",
        f"**Actionable Next Step:** {next_step}",
        "",
    ]
    if strongest_areas:
        export_lines.append("## Top Demonstrated Strengths")
        for sa in strongest_areas:
            export_lines.append(f"- {sa}")
        export_lines.append("")
    if priority_improvements:
        export_lines.append("## Priority Growth Opportunities")
        for pi in priority_improvements:
            export_lines.append(f"- {pi}")
        export_lines.append("")

    export_content = "\n".join(export_lines)
    export_filename = f"{candidate_name.replace(' ', '_')}_Interview_Report.md"

    col_act1, col_act2, col_act3 = st.columns([2, 2, 2])
    with col_act1:
        if st.button("➕ Start New Practice Interview", type="primary", use_container_width=True):
            reset_interview_state()
            st.rerun()
    with col_act2:
        if st.button("📋 View All Interview History", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    with col_act3:
        st.download_button(
            label="📄 Export Report (.md)",
            data=export_content,
            file_name=export_filename,
            mime="text/markdown",
            use_container_width=True,
        )
