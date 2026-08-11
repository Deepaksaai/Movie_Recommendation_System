from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from app.components.movie_card import render_movie_card


def render_recommendation_section(
    title: str,
    movies: List[Dict[str, Any]],
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
    empty_message: str = "Nothing to show yet.",
):

    st.subheader(title)

    if subtitle:
        st.caption(subtitle)

    if not movies:
        st.info(empty_message)
        return

    NUM_COLS = 4
    cols = st.columns(NUM_COLS)

    for i, movie in enumerate(movies):
        with cols[i % NUM_COLS]:
            render_movie_card(movie, badge)
