"""
inference.py

Run inference using the trained PPO recommender.
"""

import pandas as pd
from stable_baselines3 import PPO

from environment import MovieRecommendationEnv


# ---------------------------------------------------
# Load MovieLens titles
# ---------------------------------------------------

movies = pd.read_csv("movies.csv")

movie_lookup = dict(zip(movies.movieId, movies.title))


def title(movie_id):
    return movie_lookup.get(int(movie_id), f"Movie {movie_id}")


# ---------------------------------------------------
# Load Environment
# ---------------------------------------------------

env = MovieRecommendationEnv(
    device="cuda"
)

# ---------------------------------------------------
# Load PPO
# ---------------------------------------------------

model = PPO.load(
    "trained_models/ppo_final",
    env=env,
    device="cuda"
)

# ---------------------------------------------------
# Evaluate multiple users
# ---------------------------------------------------

NUM_USERS = 10

available_users = env.history_store.user_ids

print(f"\nFound {len(available_users)} users with valid embeddings.\n")

for idx, user_id in enumerate(available_users[:NUM_USERS], start=1):

    obs, info = env.reset(
        options={
            "user_id": int(user_id)
        }
    )

    print("\n")
    print("=" * 80)
    print(f"USER {idx}/{NUM_USERS}  (MovieLens User ID = {user_id})")
    print("=" * 80)

    print("\nHistory")
    print("-" * 80)

    history = info["history"][-10:]

    for movie in history:
        print(title(movie))

    print("\nTop-5 Greedy Recommendations")
    print("-" * 80)

    for i, movie in enumerate(info["greedy_ids"], start=1):
        print(f"{i:2d}. {title(movie)}")

    print("\nPPO Recommendations")
    print("-" * 80)

    terminated = False
    truncated = False

    ppo_movies = []

    rank = 6

    while not (terminated or truncated):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        movie_id = int(env.candidate_ids[action])

        if movie_id not in ppo_movies:
            ppo_movies.append(movie_id)

            print(f"{rank:2d}. {title(movie_id)}")

            rank += 1

        obs, reward, terminated, truncated, step_info = env.step(action)

    print("\nHoldout Movies")
    print("-" * 80)

    for movie in info["holdout"]:
        print(title(movie))

    print("\nFinal Recommendation List")
    print("-" * 80)

    final_list = list(info["greedy_ids"]) + ppo_movies

    for i, movie in enumerate(final_list, start=1):
        print(f"{i:2d}. {title(movie)}")

    print("\n")