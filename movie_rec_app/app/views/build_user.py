"""
app/views/build_user.py
=========================
Mode 2 -- "Build Your Own User": a visitor searches movies.csv, assembles
a small watch history, and asks the backend for cold-start recommendations.

The backend method for this (`recommend_from_history`) doesn't exist yet.
This page is fully functional up to that point and shows a "coming soon"
message instead of fabricating results -- swapping in the real method later
only requires backend work in `app/utils/backend.py::recommend_from_selection`.
"""

from __future__ import annotations

import streamlit as st

from app.components.movie_card import render_movie_card
from app.components.recommendation_grid import render_recommendation_section
from app.components.search_box import (
    clear_movies,
    get_selected_movie_ids,
    remove_movie,
    render_search_box,
)
from app.utils.backend import recommend_from_selection
from app.utils.data import movies_by_ids

MIN_RECOMMENDED_MOVIES = 3


def render_build_user_page(movies_df) -> None:

    st.markdown("## 🧑‍🚀 Build Your Own User")
    st.caption(
        "Simulate being a brand-new visitor: search for movies you like, add them to "
        "your history, then generate cold-start recommendations."
    )

    st.markdown("#### Search movies")
    render_search_box(movies_df, key_prefix="build_user_search")

    st.divider()

    selected_ids = get_selected_movie_ids()
    st.markdown(f"#### My Movies ({len(selected_ids)})")

    if not selected_ids:
        st.info("You haven't added any movies yet. Search above and click **Add**.")
    else:
        selected_df = movies_by_ids(movies_df, selected_ids)
        # preserve the order the user added them in
        selected_df = selected_df.set_index("movieId").loc[selected_ids].reset_index()

        cols = st.columns(min(5, len(selected_df)) or 1)
        for i, (_, row) in enumerate(selected_df.iterrows()):
            with cols[i % len(cols)]:
                card_html = render_movie_card(
                    {
                        "movie_id": int(row["movieId"]),
                        "title": row["title"],
                        "genres": row["genres"],
                        "tmdb_id": row["tmdb_id"] if "tmdb_id" in row else None,
                    }
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{row['movieId']}", use_container_width=True):
                    remove_movie(int(row["movieId"]))
                    st.rerun()

        st.button("Clear all", on_click=clear_movies)

    st.divider()

    can_generate = len(selected_ids) >= MIN_RECOMMENDED_MOVIES
    if not can_generate and selected_ids:
        st.caption(
            f"Add at least {MIN_RECOMMENDED_MOVIES} movies for better recommendations "
            f"({len(selected_ids)}/{MIN_RECOMMENDED_MOVIES})."
        )

    generate = st.button(
        "Generate Recommendations",
        type="primary",
        disabled=not selected_ids,
        use_container_width=True,
    )

    result_key = "build_user_result"

    if generate:
        with st.spinner("Generating cold-start recommendations..."):
            st.session_state[result_key] = recommend_from_selection(selected_ids)
            st.session_state["build_user_result_pending"] = False

    result = st.session_state.get(result_key)

    if generate or result_key in st.session_state:
        if result is None:
            st.warning(
                "implement before college starts"
            )
        else:
            render_recommendation_section(
                "Top 5 Greedy Recommendations",
                result.get("greedy", []),
                subtitle="Highest-scoring candidates for this new user profile.",
                badge="Greedy",
            )
            render_recommendation_section(
                "Top PPO Recommendations",
                result.get("ppo", []),
                subtitle="Re-ranked by the trained PPO policy.",
                badge="PPO",
            )
