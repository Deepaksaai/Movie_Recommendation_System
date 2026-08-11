"""
app/app.py
==========
Main Streamlit entry point.

Run with:
    streamlit run app/app.py

Layout: sidebar navigation (no tabs) between the two modes described in the
project spec -- "Existing User Demo" and "Build Your Own User".
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `from app.xxx import ...` regardless of the working directory this
# is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.views.build_user import render_build_user_page  # noqa: E402
from app.views.existing_user import render_existing_user_page  # noqa: E402
from app.utils.data import load_movies  # noqa: E402

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_THEME_CSS = """
<style>
:root {
    --app-bg: #0e0f13;
    --app-surface: #16181d;
    --app-border: #262a33;
    --app-accent: #e50914;
    --app-text: #f2f2f2;
    --app-text-muted: #9a9ea6;
}

.stApp {
    background-color: var(--app-bg);
}

section[data-testid="stSidebar"] {
    background-color: #101115;
    border-right: 1px solid var(--app-border);
}

h1, h2, h3, h4 {
    color: var(--app-text) !important;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--app-border);
    background-color: #1c1f26;
    color: var(--app-text);
    font-weight: 600;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--app-accent);
    color: var(--app-accent);
}
.stButton > button[kind="primary"] {
    background-color: var(--app-accent);
    border-color: var(--app-accent);
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background-color: #b70710;
    border-color: #b70710;
    color: white;
}

.app-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 4px;
}
.app-header .title {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--app-text);
}
.app-header .subtitle {
    color: var(--app-text-muted);
    font-size: 0.9rem;
}

div[data-testid="stTextInput"] input {
    background-color: #1c1f26;
    border: 1px solid var(--app-border);
    color: var(--app-text);
}

hr {
    border-color: var(--app-border) !important;
}
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

PAGES = {
    "Existing User Demo": "existing_user",
    "Build Your Own User": "build_user",
}


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🎬 Movie Recommender")
        st.caption("RL-based recommendations · MovieLens 32M")
        st.markdown("---")
        choice = st.radio(
            "Navigate",
            options=list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(
            "Pipeline: Transformer User Encoder → FAISS Retrieval → "
            "Greedy Top-5 → PPO Re-ranking"
        )
    return PAGES[choice]


def main() -> None:
    st.markdown(
        """
        <div class="app-header">
            <span class="title">Movie Recommendation System</span>
            <span class="subtitle">Reinforcement-Learning powered recommendations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = render_sidebar()

    if page == "existing_user":
        render_existing_user_page()
    elif page == "build_user":
        movies_df = load_movies()
        render_build_user_page(movies_df)


if __name__ == "__main__":
    main()
