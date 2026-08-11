"""
app/utils/backend.py
=====================
The ONLY place in the frontend that touches `backend.recommender`.

Everything else in `app/` calls `get_recommender()`, `recommend_existing()`
or `recommend_from_selection()` -- never the MovieRecommender class
directly. This keeps the backend a black box and means that when
`recommend_from_history()` lands for real, only `recommend_from_selection()`
below needs to change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Make the sibling `backend/` package importable without touching it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.recommender import MovieRecommender  # noqa: E402


@st.cache_resource(show_spinner="Loading recommender model...")
def get_recommender(device: str = "cpu") -> MovieRecommender:
    """
    Load the MovieRecommender exactly once per server process (cached as a
    resource, not data, since it likely holds a model / FAISS index / GPU
    tensors). Change `device` to "cuda" wherever this is actually deployed
    with a GPU available.
    """
    return MovieRecommender(device=device)


def recommend_existing(user_id: int) -> Dict[str, Any]:
    """Mode 1: existing MovieLens user -> full recommendation bundle."""
    rec = get_recommender()
    return rec.recommend(user_id)


def recommend_from_selection(movie_ids: List[int]) -> Optional[Dict[str, Any]]:
    """
    Mode 2: brand-new / cold-start user built from manually selected movies.

    This calls `recommend_from_history()` if the backend has implemented it,
    and returns None otherwise so the UI can show a "coming soon" message
    instead of inventing fake recommendations.

    *** When the real backend adds `recommend_from_history`, no changes are
    needed here or anywhere else in the app -- this function already wires
    it up. ***
    """
    rec = get_recommender()
    if not hasattr(rec, "recommend_from_history"):
        return None

    result = rec.recommend_from_history(movie_ids)
    return result  # may legitimately be None if not implemented yet
