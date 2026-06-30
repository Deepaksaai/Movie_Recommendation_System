import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

RATINGS_PATH = "ratings.csv"
MOVIE_IDS_PATH = "trained_movie_encoder/movie_ids.npy"
OUTPUT_PATH = "transformer_users.pkl"

POS_THRESHOLD = 4.0
NEG_THRESHOLD = 2.0
MIN_HISTORY = 2

random.seed(42)

# ==========================================================
# LOAD
# ==========================================================

ratings = pd.read_csv(
    RATINGS_PATH,
    usecols=[
        "userId",
        "movieId",
        "rating",
        "timestamp"
    ]
)

movie_ids = set(np.load(MOVIE_IDS_PATH))

ratings = ratings[
    ratings.movieId.isin(movie_ids)
]

ratings = ratings.sort_values(
    ["userId", "timestamp"]
)

# ==========================================================
# BUILD USER DATASET
# ==========================================================

users = []

groups = ratings.groupby("userId")

for user_id, group in tqdm(groups):

    positives = group[
        group.rating >= POS_THRESHOLD
    ]["movieId"].tolist()

    negatives = group[
        group.rating <= NEG_THRESHOLD
    ]["movieId"].tolist()

    if len(positives) < MIN_HISTORY + 1:
        continue

    if len(negatives) == 0:
        continue

    users.append({

        "user_id": int(user_id),

        "positives": positives,

        "negatives": negatives

    })

print()
print(f"Users Saved : {len(users)}")

avg_pos = np.mean([len(u["positives"]) for u in users])
avg_neg = np.mean([len(u["negatives"]) for u in users])

print(f"Average Positive Movies : {avg_pos:.2f}")
print(f"Average Negative Movies : {avg_neg:.2f}")

with open(OUTPUT_PATH, "wb") as f:

    pickle.dump(users, f)

print(f"Saved : {OUTPUT_PATH}")