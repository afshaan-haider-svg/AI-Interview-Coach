"""Custom modern CSS styles for the AI Interview Coach Streamlit application."""

CUSTOM_CSS = """
<style>
/* Base typography and container tweaks */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0F172A;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 2.5rem 2rem;
    border-radius: 1rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.2);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
    color: #FFFFFF !important;
}

.hero-subtitle {
    font-size: 1.15rem;
    opacity: 0.92;
    margin-bottom: 1.5rem;
    max-width: 650px;
    line-height: 1.6;
    color: #E0E7FF !important;
}

/* Feature Cards */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 0.75rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
}

.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.feature-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.25rem;
}

.feature-desc {
    font-size: 0.9rem;
    color: #64748B;
    line-height: 1.5;
}

/* Question Box */
.question-card {
    background: #FFFFFF;
    border-left: 5px solid #4F46E5;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}

.question-meta {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-easy { background-color: #DCFCE7; color: #166534; }
.badge-medium { background-color: #FEF3C7; color: #92400E; }
.badge-hard { background-color: #FEE2E2; color: #991B1B; }
.badge-track { background-color: #EEF2FF; color: #3730A3; }
.badge-category { background-color: #F1F5F9; color: #475569; }

/* Metric and Score Cards */
.metric-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #4F46E5;
}

.metric-label {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 500;
    margin-top: 0.25rem;
}

/* Adaptive Banner */
.adaptive-banner {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    color: #1E40AF;
}

/* Topic Chips */
.topic-chip {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 0.375rem;
    font-size: 0.8rem;
    margin: 0.2rem;
    font-weight: 500;
}

.topic-covered { background-color: #DCFCE7; color: #15803D; }
.topic-missing { background-color: #FEE2E2; color: #B91C1C; }

/* Clean UI utility */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    border-bottom: 2px solid #F1F5F9;
    padding-bottom: 0.4rem;
}
</style>
"""


def apply_custom_styles():
    """Injects custom CSS styling into the Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
