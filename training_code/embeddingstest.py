import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# PATHS
# ==========================================================

EMBEDDINGS_PATH = "trained_movie_encoder/movie_embeddings.npy"
MOVIE_IDS_PATH = "trained_movie_encoder/movie_ids.npy"
MOVIES_CSV = "movies.csv"

TOP_K = 10

# ==========================================================
# LOAD
# ==========================================================

embeddings = np.load(EMBEDDINGS_PATH)
movie_ids = np.load(MOVIE_IDS_PATH)

movies = pd.read_csv(MOVIES_CSV)

movieid_to_index = {
    mid: idx
    for idx, mid in enumerate(movie_ids)
}

movieid_to_title = dict(
    zip(movies.movieId, movies.title)
)

# ==========================================================
# SEARCH FUNCTION
# ==========================================================

def search(movie_name):

    matches = movies[
        movies.title.str.contains(
            movie_name,
            case=False,
            na=False
        )
    ]

    if len(matches) == 0:
        print("Movie not found.")
        return

    movie = matches.iloc[0]

    movie_id = movie.movieId

    if movie_id not in movieid_to_index:
        print("Movie has no embedding.")
        return

    idx = movieid_to_index[movie_id]

    query = embeddings[idx].reshape(1, -1)

    sims = cosine_similarity(
        query,
        embeddings
    )[0]

    order = np.argsort(-sims)

    print("=" * 60)
    print("Query :", movie.title)
    print("=" * 60)

    count = 0

    for i in order:

        if movie_ids[i] == movie_id:
            continue

        title = movieid_to_title.get(
            movie_ids[i],
            "Unknown"
        )

        print(f"{count+1:2d}. {title:60} {sims[i]:.4f}")

        count += 1

        if count == TOP_K:
            break

# ==========================================================
# INTERACTIVE
# ==========================================================

while True:

    movie = input("\nMovie (or quit): ")

    if movie.lower() == "quit":
        break

    search(movie)