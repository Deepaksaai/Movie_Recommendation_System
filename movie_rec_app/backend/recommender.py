# recommender.py

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from rl.environment import MovieRecommendationEnv


class MovieRecommender:
    def __init__(
        self,
        device: str = "cpu",
        movies_path: str = "movies.csv",
        links_path: str = "links.csv",
        user_ids_path: str = "trained_movie_encoder/user_ids.npy",
        ppo_model_path: str = "trained_models/ppo_final.zip",
    ):
        self.device = device

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------
        self.movies_df = pd.read_csv(movies_path)
        self.links_df = pd.read_csv(links_path)
        self.ratings_df = pd.read_csv("ratings.csv")

        # Merge TMDB ids into movie metadata
        metadata = self.movies_df.merge(
            self.links_df[["movieId", "tmdbId"]],
            on="movieId",
            how="left",
        )

        metadata["tmdbId"] = metadata["tmdbId"].fillna(-1).astype(int)

        # --------------------------------------------------
        # Movie lookup dictionary
        # movie_id -> metadata
        # --------------------------------------------------
        self.movie_lookup = {}

        for row in metadata.itertuples(index=False):
            self.movie_lookup[int(row.movieId)] = {
                "movie_id": int(row.movieId),
                "title": row.title,
                "genres": row.genres,
                "tmdb_id": None if row.tmdbId == -1 else int(row.tmdbId),
            }

        # --------------------------------------------------
        # Available users
        # --------------------------------------------------
        self.available_users = np.load(user_ids_path).astype(int)

        # --------------------------------------------------
        # Create recommendation environment
        # --------------------------------------------------
        self.env = MovieRecommendationEnv(device=device)

        # --------------------------------------------------
        # Load PPO model
        # --------------------------------------------------
        self.model = PPO.load(
            ppo_model_path,
            env=self.env,
            device=device,
        )

    def _movie_info(self, movie_id: int) -> dict:
        """
        Convert a MovieLens movie ID into metadata.
        """
        movie_id = int(movie_id)

        if movie_id in self.movie_lookup:
            return self.movie_lookup[movie_id].copy()

        return {
            "movie_id": movie_id,
            "title": f"Unknown ({movie_id})",
            "genres": "",
            "tmdb_id": None,
        }


    def recommend(self, user_id: int):
        """
        Returns history, greedy recommendations, PPO recommendations,
        and holdout movies.
        """

        obs, info = self.env.reset(
            options={"user_id": int(user_id)}
        )

        # --------------------------------------------------
        # PPO Recommendations
        # --------------------------------------------------
        ppo_movies = []
        seen = set(int(mid) for mid in info["greedy_ids"])

        terminated = False
        truncated = False

        while (
            len(ppo_movies) < 5
            and not terminated
            and not truncated
        ):
            action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, step_info = self.env.step(
                int(action)
            )

            movie_id = int(step_info["movie_id"])

            if (
                step_info["already_recommended"]
                or movie_id in seen
            ):
                continue

            seen.add(movie_id)
            ppo_movies.append(self._movie_info(movie_id))

        result = {
            "user_id": int(user_id),

            "history": self._top_rated_movies(user_id),

            "greedy": [
                self._movie_info(mid)
                for mid in info["greedy_ids"]
            ],

            "ppo": ppo_movies,

            "holdout": [
                self._movie_info(mid)
                for mid in info["holdout"]
            ],
        }

        return result

    def _top_rated_movies(self, user_id: int, n: int = 10):
        ratings = (
            self.ratings_df[self.ratings_df["userId"] == user_id]
            .sort_values(["rating", "timestamp"], ascending=[False, False])
            .head(n)
        )

        movies = []

        for row in ratings.itertuples(index=False):
            movie = self._movie_info(row.movieId)
            movie["rating"] = float(row.rating)
            movies.append(movie)

        return movies


        print("=" * 60)
        print("MovieRecommender initialized successfully.")
        print(f"Users loaded      : {len(self.available_users):,}")
        print(f"Movies loaded     : {len(self.movie_lookup):,}")
        print(f"Device            : {device}")
        print("=" * 60)