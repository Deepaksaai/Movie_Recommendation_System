"""
app/components/search_box.py
==============================
Search widget + result rows with "Add" buttons, used by the
"Build Your Own User" page to let a visitor assemble a movie history.

Selected movies live in `st.session_state["my_movies"]` as a list of
movie_id ints, so other components (the "My Movies" tray, the generate
button) can read/write the same state.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils.data import search_movies

SESSION_KEY = "my_movies"


def init_my_movies_state() -> None:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = []  # list[int] movie_id, in add order


def get_selected_movie_ids() -> list[int]:
    init_my_movies_state()
    return st.session_state[SESSION_KEY]


def add_movie(movie_id: int) -> None:
    init_my_movies_state()
    if movie_id not in st.session_state[SESSION_KEY]:
        st.session_state[SESSION_KEY].append(movie_id)


def remove_movie(movie_id: int) -> None:
    init_my_movies_state()
    if movie_id in st.session_state[SESSION_KEY]:
        st.session_state[SESSION_KEY].remove(movie_id)


def clear_movies() -> None:
    st.session_state[SESSION_KEY] = []


def render_search_box(movies_df: pd.DataFrame, key_prefix: str = "search") -> None:
    """
    Renders the search input and result rows with Add buttons.
    Adds/removes directly mutate `st.session_state["my_movies"]`.
    """
    init_my_movies_state()

    query = st.text_input(
        "Search movies",
        placeholder="Search by title or year, e.g. 'inter' or '2010'",
        key=f"{key_prefix}_query",
        label_visibility="collapsed",
    )

    if not query.strip():
        st.caption("Start typing a title or year to find movies.")
        return

    results = search_movies(movies_df, query, limit=12)

    if results.empty:
        st.caption(f"No movies found for “{query}”.")
        return

    selected_ids = set(get_selected_movie_ids())

    for _, row in results.iterrows():
        movie_id = int(row["movieId"])
        already_added = movie_id in selected_ids

        col_info, col_action = st.columns([5, 1], vertical_alignment="center")
        with col_info:
            year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
            genres = str(row.get("genres", "")).replace("|", " · ")
            st.markdown(
                f"**{row['title_clean']}**{year}  \n"
                f"<span style='color:#8a8f99; font-size:0.82rem;'>{genres}</span>",
                unsafe_allow_html=True,
            )
        with col_action:
            if already_added:
                st.button("Added", key=f"{key_prefix}_added_{movie_id}", disabled=True, use_container_width=True)
            else:
                if st.button("Add", key=f"{key_prefix}_add_{movie_id}", use_container_width=True):
                    add_movie(movie_id)
                    st.rerun()
