import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

POSTER_BASE = "https://image.tmdb.org/t/p/w500"

PLACEHOLDER = "https://placehold.co/500x750?text=No+Poster"


@st.cache_data(show_spinner=False)
def get_poster(tmdb_id):

    if tmdb_id is None:
        return PLACEHOLDER

    if API_KEY is None:
        return PLACEHOLDER

    try:
        url = (
            f"https://api.themoviedb.org/3/movie/"
            f"{tmdb_id}"
            f"?api_key={API_KEY}"
        )

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return PLACEHOLDER

        data = response.json()

        poster = data.get("poster_path")

        if poster is None:
            return PLACEHOLDER

        return POSTER_BASE + poster

    except Exception:
        return PLACEHOLDER