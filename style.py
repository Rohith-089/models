
import streamlit as st


# Color palette
ACCENT = "#2563EB"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E5E7EB"


# CSS styles
CSS = f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');


/* 1. Font family */

.stApp,
.stMarkdown,
p,
label,
h1,
h2,
h3,
h4,
button,
input,
textarea,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"] {{
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}


/* 2. Hide default Streamlit elements */

#MainMenu,
footer {{
    visibility: hidden;
}}

/* Dotted background pattern */

.stApp {{
    background-color: #F5F5F5;
    background-image: radial-gradient(
        circle,
        #B8B8B8 1.3px,
        transparent 1.5px
    );
    background-size: 152px 152px;
    background-position: 82px 24px;
    background-attachment: fixed;
}}


/* Keep the background transparent so the dots remain visible */

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main {{
    background-color: transparent;
}}



/* 3. Main page layout */

.block-container {{
    max-width: 1080px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}}


/* 4. Main heading */

h1 {{
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 1.8rem;
    color: {INK};
}}


/* 5. Smaller heading */

h4 {{
    font-weight: 600;
    letter-spacing: -0.01em;
    color: {INK};
}}


/* 6. Step label: Step 1, Step 2, etc. */

.step {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {ACCENT};
    margin: 2.2rem 0 0.1rem;
}}


/* 7. Step title: Choose a model, etc. */

.step-title {{
    font-size: 1.25rem;
    font-weight: 600;
    color: #D9361E;

    margin-bottom: 0.2rem;
}}


/* 8. Step description */

.step-desc {{
    color: {MUTED};
    font-size: 0.92rem;
    margin-bottom: 0.8rem;
}}


/* 9. Metric cards */

[data-testid="stMetric"] {{

    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
}}


/* 10. Metric labels */

[data-testid="stMetricLabel"] {{
    color: {MUTED};
}}


/* 11. Metric values */

[data-testid="stMetricValue"] {{
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}


/* 12. Primary buttons */

.stButton > button[kind="primary"] {{
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.4rem;
}}


/* 13. Download buttons */

.stDownloadButton > button {{
    border-radius: 8px;
    font-weight: 500;
}}


/* 14. File uploader */

[data-testid="stFileUploader"] section {{
    border-radius: 10px;
    ;
}}


/* 15. Tabs */

.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
}}

</style>
"""


# Apply CSS to the Streamlit app
def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# Display a numbered step with an optional description
def step(n, title, desc=""):
    html = (
        f'<div class="step">Step {n}</div>'
        f'<div class="step-title">{title}</div>'
    )

    if desc:
        html += f'<div class="step-desc">{desc}</div>'

    st.markdown(html, unsafe_allow_html=True)
