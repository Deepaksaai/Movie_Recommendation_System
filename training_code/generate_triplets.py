import os
import random
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------

RATINGS_PATH = "ratings.csv"

OUTPUT_PATH = "triplets.csv"

POS_THRESHOLD = 4.0
NEG_THRESHOLD = 2.0

MAX_POS_PER_ANCHOR = 3
MAX_NEG_PER_ANCHOR = 3

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# -----------------------------
# LOAD RATINGS
# -----------------------------

print("Loading ratings...")

ratings = pd.read_csv(
    RATINGS_PATH,
    usecols=["userId", "movieId", "rating"]
)

print(ratings.shape)

# -----------------------------
# BUILD USER HISTORIES
# -----------------------------

print("Grouping users...")

user_groups = ratings.groupby("userId")

triplets = []

# -----------------------------
# GENERATE TRIPLETS
# -----------------------------

print("Generating triplets...")

for user_id, group in tqdm(user_groups):

    positives = group[group.rating >= POS_THRESHOLD]["movieId"].tolist()

    negatives = group[group.rating <= NEG_THRESHOLD]["movieId"].tolist()

    if len(positives) < 2:
        continue

    if len(negatives) == 0:
        continue

    for anchor in positives:

        positive_candidates = [
            x for x in positives
            if x != anchor
        ]

        if len(positive_candidates) == 0:
            continue

        sampled_pos = random.sample(
            positive_candidates,
            min(MAX_POS_PER_ANCHOR,
                len(positive_candidates))
        )

        sampled_neg = random.sample(
            negatives,
            min(MAX_NEG_PER_ANCHOR,
                len(negatives))
        )

        for pos in sampled_pos:

            for neg in sampled_neg:

                triplets.append(
                    (
                        anchor,
                        pos,
                        neg
                    )
                )

# -----------------------------
# SAVE
# -----------------------------

triplets = pd.DataFrame(
    triplets,
    columns=[
        "anchor",
        "positive",
        "negative"
    ]
)

triplets.drop_duplicates(inplace=True)

triplets.to_csv(
    OUTPUT_PATH,
    index=False
)

print()

print("Triplets generated:", len(triplets))

print(triplets.head())

