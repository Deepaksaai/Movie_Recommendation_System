import faiss
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

INDEX_PATH = "trained_movie_encoder/movie_index.faiss"

USER_EMBEDDINGS = "trained_movie_encoder/user_embeddings.npy"
USER_IDS = "trained_movie_encoder/user_ids.npy"

MOVIE_IDS = "trained_movie_encoder/movie_ids.npy"

MOVIES_CSV = "movies.csv"

TOP_K = 10

# ==========================================================
# LOAD
# ==========================================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print("Loading embeddings...")

user_embeddings = np.load(USER_EMBEDDINGS).astype(np.float32)
user_ids = np.load(USER_IDS)

movie_ids = np.load(MOVIE_IDS)

movies = pd.read_csv(MOVIES_CSV)

ratings = pd.read_csv(
    "ratings.csv",
    usecols=["userId", "movieId", "rating", "timestamp"]
)

ratings = ratings.sort_values(["userId", "timestamp"])

movie_title = dict(
    zip(
        movies.movieId,
        movies.title
    )
)

# ==========================================================
# USER
# ==========================================================

print()

user = int(input("Enter User ID: "))

if user not in user_ids:

    print("User not found.")
    quit()

idx = np.where(user_ids == user)[0][0]

query = user_embeddings[idx].reshape(1, -1)

# ----------------------------------------------------------
# Show user's recent positive history
# ----------------------------------------------------------

history = ratings[
    (ratings.userId == user) &
    (ratings.rating >= 4.0)
]

history = history.tail(10)

print()
print("=" * 70)
print(f"User {user} - Recent Positive History")
print("=" * 70)

for i, (_, row) in enumerate(history.iterrows(), start=1):

    title = movie_title.get(
        row.movieId,
        "Unknown"
    )

    print(f"{i:2d}. {title}")

# ==========================================================
# SEARCH
# ==========================================================

scores, indices = index.search(query, TOP_K)

print()
print("=" * 70)
print(f"Top {TOP_K} recommendations for User {user}")
print("=" * 70)

for rank, (score, movie_index) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    movie_id = int(movie_ids[movie_index])

    title = movie_title.get(
        movie_id,
        "Unknown"
    )

    print(
        f"{rank:2d}. {title:60} {score:.4f}"
    )