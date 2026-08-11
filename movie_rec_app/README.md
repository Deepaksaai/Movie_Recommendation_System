# Movie Recommendation System — Streamlit Frontend

A Streamlit UI for the RL-based movie recommender (Transformer user encoder →
FAISS retrieval → Greedy top-5 → PPO re-ranking). This repo contains **only
the frontend** — the trained pipeline is treated as a black box.

## Project structure

```
app/
    app.py                     # entry point: sidebar nav, dark theme, routing
    views/
        existing_user.py       # Mode 1: Existing User Demo
        build_user.py          # Mode 2: Build Your Own User
    components/
        movie_card.py          # single Netflix-style movie card (HTML/CSS)
        recommendation_grid.py # titled section + responsive card grid
        search_box.py          # movie search + Add buttons + "My Movies" state
    utils/
        tmdb.py                 # get_poster(tmdb_id), cached, with placeholder fallback
        data.py                 # movies.csv loading + title/year search
        backend.py               # the ONLY file that imports backend.recommender
backend/
    recommender.py             # *** PLACEHOLDER *** — replace with your real class
data/
    movies.csv                 # small sample dataset for local dev/demo
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Wiring up your real backend

1. Drop your real, trained `backend/recommender.py` in, replacing the
   stub. Keep the class name `MovieRecommender` and the `recommend(user_id)`
   signature identical — nothing else needs to change.
2. If your app needs a real "list of existing user ids" for the dropdown
   (instead of the stub's `list_demo_user_ids()`), either add that method
   to `MovieRecommender` too, or hardcode/load a list in
   `app/views/existing_user.py`.
3. **Cold start (`recommend_from_history`)**: the moment your backend
   implements `MovieRecommender.recommend_from_history(movie_ids)`, the
   "Build Your Own User" page will start showing real recommendations
   automatically — the wiring already lives in
   `app/utils/backend.py::recommend_from_selection()` and checks for the
   method via `hasattr`. No other file needs to change.
4. Swap `device="cpu"` for `device="cuda"` in
   `app/utils/backend.py::get_recommender()` wherever you deploy with a GPU.

## TMDB posters

Set a TMDB v3 API key as either:

- a Streamlit secret: `.streamlit/secrets.toml` → `TMDB_API_KEY = "..."`, or
- an environment variable: `TMDB_API_KEY=...`

Posters are cached (`st.cache_data`, 24h TTL) per `tmdb_id`. If no key is
configured, or a poster can't be found, `get_poster()` falls back to a
neutral placeholder image automatically — the UI never breaks because of a
missing poster.

## `data/movies.csv`

A small (68-title) hand-picked sample is included so the app is runnable
out of the box. Swap in your real MovieLens `movies.csv` — the loader in
`app/utils/data.py` expects at minimum `movieId, title, genres` columns and
will derive `year` from the title (`"Movie Title (1999)"`) if a `year`
column isn't present, and treats `tmdb_id` as optional (missing values are
handled gracefully by the poster fallback).

## Notes on the current stub backend

`backend/recommender.py` in this repo is **not** the real ML pipeline — it's
a small deterministic stand-in (samples from `movies.csv`) so the UI is
demoable and testable end to end before the real model is dropped in. It
implements the exact same interface described in the project spec, so
swapping it out is a drop-in replacement. `recommend_from_history()`
intentionally returns `None` to reflect that it isn't implemented in the
real backend yet — the frontend already handles that by showing a
"Coming soon" message instead of fabricating results.
