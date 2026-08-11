"""
app/utils/data.py
==================
Loading and searching movies.csv. Read-only, cached, no backend/model logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "movies.csv"


@st.cache_data(show_spinner=False)
def load_movies(path: Optional[str] = None) -> pd.DataFrame:
    """Load movies.csv once and cache it for the whole session."""
    csv_path = Path(path) if path else _DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path)

    # Best-effort year extraction if a "year" column isn't already present
    if "year" not in df.columns:
        df["year"] = df["title"].apply(_extract_year)
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "tmdb_id" in df.columns:
        df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")

    df["title_clean"] = df["title"].apply(_strip_year)
    return df


def _extract_year(title: str) -> Optional[int]:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def _strip_year(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def search_movies(df: pd.DataFrame, query: str, limit: int = 25) -> pd.DataFrame:
    """
    Search by partial title match or by a 4-digit year.
    Case-insensitive, substring based -- e.g. 'inter' matches 'Interstellar'.
    """
    query = (query or "").strip()
    if not query:
        return df.iloc[0:0]

    if re.fullmatch(r"\d{4}", query):
        mask = df["year"] == int(query)
        return df[mask].sort_values("title_clean").head(limit)

    pattern = re.escape(query)
    mask = df["title_clean"].str.contains(pattern, case=False, na=False, regex=True)
    results = df[mask].copy()

    # Rank matches that start with the query above ones that merely contain it
    starts_with = results["title_clean"].str.lower().str.startswith(query.lower())
    results["_rank"] = (~starts_with).astype(int)
    results = results.sort_values(["_rank", "title_clean"]).drop(columns="_rank")

    return results.head(limit)


def movies_by_ids(df: pd.DataFrame, movie_ids: list[int]) -> pd.DataFrame:
    return df[df["movieId"].isin(movie_ids)]
