from __future__ import annotations

import re
from typing import Any, Dict

import streamlit as st

from app.utils.tmdb import get_poster


def _split_title_year(title: str):
    match = re.search(r"^(.*?)\s*\((\d{4})\)\s*$", title)
    if match:
        return match.group(1), match.group(2)
    return title, ""


def _genres(genres: str):
    if not genres or genres == "(no genres listed)":
        return "Genres unavailable"
    return genres.replace("|", " • ")


def render_movie_card(movie: Dict[str, Any], badge: str | None = None):
    """
    Render one movie card using native Streamlit widgets.
    """

    title, year = _split_title_year(movie.get("title", "Unknown"))

    poster = get_poster(movie.get("tmdb_id"))

    st.image(
        poster,
        use_container_width=True,
    )

    if badge:
        st.markdown(
            f"<span style='color:#E50914;font-weight:bold'>{badge}</span>",
            unsafe_allow_html=True,
        )

    rating = movie.get("rating")
    if rating is not None:
        stars = "⭐" * int(rating)
        st.markdown(f"**{stars} {rating:.1f}/5**")

    st.markdown(f"**{title}**")

    if year:
        st.caption(year)

    st.caption(_genres(movie.get("genres", "")))