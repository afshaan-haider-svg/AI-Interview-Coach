"""Home page component featuring professional hero banner and key capabilities."""

import streamlit as st


def render_home_page():
    """Renders the landing/home screen with hero banner, feature pillars, and CTAs."""
    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">AI-Powered Interview Coach</div>
            <div class="hero-subtitle">
                Practice smarter with personalized, adaptive AI interviews grounded in your actual experience and target job requirements.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary Action Buttons
    col_cta1, col_cta2, _ = st.columns([1.5, 1.5, 2])
    with col_cta1:
        if st.button("🚀 Start New Interview", type="primary", use_container_width=True):
            st.session_state.current_page = "setup"
            st.rerun()
    with col_cta2:
        if st.button("📋 View Interview History", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 🌟 Key Capabilities")

    # Feature Grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Resume & JD Grounding</div>
                <div class="feature-desc">
                    Upload your resume and target job description to practice questions tailored directly to your background and required tech stack.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Adaptive Difficulty Engine</div>
                <div class="feature-desc">
                    Questions dynamically adapt between Easy, Medium, and Hard based on your actual performance with strict technical guardrails.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Instant 5-Dimension Evaluation</div>
                <div class="feature-desc">
                    Receive immediate, structured feedback evaluating Relevance, Clarity, Completeness, Technical Correctness, and Structure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Comprehensive Final Analytics</div>
                <div class="feature-desc">
                    Review deterministic score statistics, topic coverage gaps, recurring strengths, and actionable personalized study roadmaps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.info(
        "💡 **Tip:** Resume and Job Description are completely optional. "
        "You can jump straight into technical or HR practice with generic questions anytime!"
    )
