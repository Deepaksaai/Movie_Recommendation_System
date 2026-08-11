"""
app/views/existing_user.py
============================
Mode 1 -- "Existing User Demo": pick a real MovieLens user id and show
their history plus Greedy and PPO recommendations from the backend.
"""

from __future__ import annotations

import streamlit as st

from app.components.recommendation_grid import render_recommendation_section
from app.utils.backend import get_recommender, recommend_existing


def render_existing_user_page() -> None:

    st.markdown("## 🎬 Existing User Demo")
    st.caption(
        "Pick a user that already exists in the MovieLens dataset and see what the "
        "trained retrieval + PPO re-ranking pipeline recommends for them."
    )

    recommender = get_recommender()
    demo_user_ids = (
        recommender.list_demo_user_ids()
        if hasattr(recommender, "list_demo_user_ids")
        else list(range(1, 21))
    )

    col_select, col_button = st.columns([3, 1], vertical_alignment="bottom")
    with col_select:
        user_id = st.selectbox("Select User", options=demo_user_ids, key="existing_user_select")
    with col_button:
        generate = st.button("Generate Recommendations", type="primary", use_container_width=True)

    st.divider()

    result_key = "existing_user_result"

    if generate:
        with st.spinner("Running retrieval + PPO re-ranking..."):
            st.session_state[result_key] = recommend_existing(int(user_id))

    result = st.session_state.get(result_key)

    if result is None:
        st.info("Select a user and click **Generate Recommendations** to see results.")
        return

    if result.get("user_id") != user_id:
        st.caption(f"Showing cached results for user {result.get('user_id')}. Click Generate to refresh.")

    render_recommendation_section(
        "Top Rated Movies",
        result.get("history", []),
        subtitle="The 10 highest-rated movies by this user in MovieLens.",
    )

    render_recommendation_section(
        "Top 5 Greedy Recommendations",
        result.get("greedy", []),
        subtitle="Highest-scoring candidates straight out of FAISS retrieval.",
        badge="Greedy",
    )

    render_recommendation_section(
        "Top PPO Recommendations",
        result.get("ppo", []),
        subtitle="Re-ranked by the trained PPO policy for long-term engagement.",
        badge="PPO",
    )

    holdout = result.get("holdout", [])
    if holdout:
        with st.expander("Holdout set (for evaluation)"):
            render_recommendation_section("Holdout", holdout, empty_message="No holdout movies.")
