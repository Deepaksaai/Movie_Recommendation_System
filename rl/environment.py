"""
rl/environment.py

Gymnasium environment for the "re-ranking" stage of the recommender pipeline:

    Tag Genome -> Autoencoder -> Triplet FT -> Movie Embeddings (64-D)
    Ratings    -> Transformer User Encoder   -> User Embeddings  (64-D)
    FAISS (IndexFlatIP) -> Top-100 candidates
    Gymnasium Env -> PPO / A2C / DQN builds a whole recommendation LIST
                     (not a single best pick) from the 100 candidates.

The RL agent never retrieves movies. FAISS has already narrowed ~34K movies
down to 100 candidates at the start of the episode. The agent's job is to
sequentially select `episode_length` (default 5) of those 100 candidates,
with each pick depending on everything picked so far -- this is a
sequential list-construction task, not a contextual bandit.

Compatible with:
    - gymnasium.Env API (reset/step return the 5-tuple Gym>=0.26 signature)
    - stable-baselines3 PPO / A2C / DQN ("MlpPolicy")

Author: generated for the user's fixed pipeline. No architectural changes to
upstream components (autoencoder, triplet loss, transformer, FAISS) are made
or suggested here.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "faiss is required for MovieRecommendationEnv. Install with "
        "`pip install faiss-cpu` (or faiss-gpu)."
    ) from e

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("pandas is required to read ratings.csv") from e


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

EMBED_DIM = 64
NUM_CANDIDATES = 100
EPISODE_LENGTH = 5  # "Recommend 5 movies ... (or configurable)"
SAFE_RECOMMENDATIONS = 5

# Reward weighting (kept as constants inside the environment, as requested,
# but overridable via the `reward_weights` constructor arg).
#
#   reward = W_RELEVANCE  * cosine(user_embedding, selected_movie_embedding)
#          - W_REDUNDANCY * max_cosine_similarity(selected, previous_picks)
#          + W_HOLDOUT    * holdout_hit
#
# A repeat pick (re-selecting an already-recommended candidate) short
# circuits all of the above and returns REPEAT_PENALTY instead.
W_RELEVANCE = 0.7
W_REDUNDANCY = 0.2   # subtracted -- see _compute_reward
W_HOLDOUT = 0.3

# Penalty applied if the agent picks a candidate it already recommended
# this episode. The action space is a plain Discrete(100) (no native
# masking for vanilla PPO/A2C/DQN), so illegal repeats are handled via
# reward shaping rather than being blocked outright. Per spec: terminate
# stays False on a repeat -- the episode just keeps going.
REPEAT_PENALTY = -1.0

# History bookkeeping
POSITIVE_RATING_THRESHOLD = 4.0
MIN_POSITIVE_RATINGS = 10     # user must have at least this many positives to be sampled
                               # (so at least ~5 remain in context after the 5-movie holdout)
NEXT_POSITIVES = 5            # number of immediate future positives held out as ground truth
MAX_CONTEXT_LEN = 100         # cap on history length fed into the transformer (must match training)

# Observation layout:
#   user_embedding            -> EMBED_DIM
#   candidate_embeddings      -> NUM_CANDIDATES * EMBED_DIM
#   recommended_mask          -> NUM_CANDIDATES
#   running_average_embedding -> EMBED_DIM   (mean of selected_movie_embeddings so far)
RL_CANDIDATES = NUM_CANDIDATES - SAFE_RECOMMENDATIONS

OBS_DIM = (
    EMBED_DIM
    + RL_CANDIDATES * EMBED_DIM
    + RL_CANDIDATES
    + EMBED_DIM
)


# --------------------------------------------------------------------------- #
# Transformer user encoder (inference-only wrapper matching the trained arch)
# --------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching the (embed_dim, dropout) signature
    used by the training model (checkpoint key: `pos_enc.pe`). Reconstructed to
    match that signature/buffer name -- I don't have the original training file,
    so if your real implementation differs (e.g. a learned embedding table
    instead of sinusoidal), load_state_dict will still complain and you'll need
    to paste the actual class in here."""

    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerUserEncoder(nn.Module):
    """
    2-layer Transformer encoder over a user's positive movie-embedding
    history, with a prepended CLS token, sinusoidal positional encoding,
    projected and L2-normalized into a 64-D user embedding.

    This matches the training-time architecture: no attention_mask is
    passed in at inference -- the padding mask is built internally from
    history length (all-real here, since the environment never pads).
    """

    def __init__(
        self,
        embed_dim=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        self.pos_enc = PositionalEncoding(
            embed_dim,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.projection = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(self, history_embeddings):

        B = history_embeddings.size(0)

        cls = self.cls_token.expand(
            B,
            -1,
            -1
        )

        x = torch.cat(
            [cls, history_embeddings],
            dim=1
        )

        padding_mask = torch.zeros(
            B,
            history_embeddings.size(1),
            dtype=torch.bool,
            device=history_embeddings.device
        )

        cls_mask = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=history_embeddings.device
        )

        mask = torch.cat(
            [cls_mask, padding_mask],
            dim=1
        )

        x = self.pos_enc(x)

        x = self.transformer(
            x,
            src_key_padding_mask=mask
        )

        cls_out = x[:, 0, :]

        user_emb = self.projection(cls_out)

        user_emb = F.normalize(
            user_emb,
            p=2,
            dim=1
        )

        return user_emb


# --------------------------------------------------------------------------- #
# Data stores
# --------------------------------------------------------------------------- #

class MovieEmbeddingStore:
    """Holds the 64-D movie embeddings and the movieId <-> row-index mapping."""

    def __init__(self, movie_embeddings_path: str, movie_ids_path: str):
        self.embeddings = np.load(movie_embeddings_path).astype(np.float32)
        self.movie_ids = np.load(movie_ids_path)
        if self.embeddings.shape[0] != self.movie_ids.shape[0]:
            raise ValueError(
                f"movie_embeddings ({self.embeddings.shape[0]}) and "
                f"movie_ids ({self.movie_ids.shape[0]}) length mismatch."
            )
        self.id_to_idx: Dict = {int(mid): i for i, mid in enumerate(self.movie_ids)}

    def get_embedding(self, movie_id: int) -> np.ndarray:
        return self.embeddings[self.id_to_idx[int(movie_id)]]

    def get_embeddings(self, movie_ids: Sequence[int]) -> np.ndarray:
        idxs = [self.id_to_idx[int(mid)] for mid in movie_ids]
        return self.embeddings[idxs]

    def has(self, movie_id: int) -> bool:
        return int(movie_id) in self.id_to_idx


class UserHistoryStore:
    """
    Builds, per user, a chronological list of "positive" movies (rating >=
    POSITIVE_RATING_THRESHOLD) from ratings.csv, then exposes a
    context/holdout split:
        context  -> fed into the Transformer to produce the user embedding
        holdout  -> "future" positives used for the holdout reward term
    """

    def __init__(
        self,
        ratings_path: str,
        movie_embedding_store: MovieEmbeddingStore,
        positive_threshold: float = POSITIVE_RATING_THRESHOLD,
        min_positive_ratings: int = MIN_POSITIVE_RATINGS,
        next_positives: int = NEXT_POSITIVES,
    ):
        df = pd.read_csv(ratings_path)
        required_cols = {"userId", "movieId", "rating"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"ratings.csv must contain columns {required_cols}")

        # Only keep movies we actually have embeddings for.
        df = df[df["movieId"].apply(movie_embedding_store.has)]
        df = df[df["rating"] >= positive_threshold]

        if "timestamp" in df.columns:
            df = df.sort_values(["userId", "timestamp"])
        else:
            df = df.sort_values(["userId"])

        self.histories: Dict[int, List[int]] = {}
        for uid, group in df.groupby("userId"):
            movie_seq = group["movieId"].astype(int).tolist()
            if len(movie_seq) >= min_positive_ratings:
                self.histories[int(uid)] = movie_seq

        if not self.histories:
            raise ValueError(
                "No users met min_positive_ratings after filtering ratings.csv. "
                "Lower MIN_POSITIVE_RATINGS or POSITIVE_RATING_THRESHOLD."
            )

        self.user_ids: List[int] = list(self.histories.keys())
        self.next_positives = next_positives

    def sample_user_id(self, rng: random.Random) -> int:
        return rng.choice(self.user_ids)

    def get_context_and_holdout(self, user_id: int) -> Tuple[List[int], List[int]]:
        history = self.histories[int(user_id)]
        # Hold out the immediate next `next_positives` movies chronologically,
        # e.g. history = [M1..M80, M81..M85] -> context = M1..M80,
        # holdout (ground truth) = M81..M85. Always leave at least 1 movie
        # in context even if the user has few ratings.
        n_holdout = min(self.next_positives, len(history) - 1)
        context, holdout = history[:-n_holdout], history[-n_holdout:]
        if len(context) > MAX_CONTEXT_LEN:
            context = context[-MAX_CONTEXT_LEN:]
        return context, holdout


class FaissRetriever:
    """Thin wrapper around a pre-built faiss.IndexFlatIP over movie embeddings."""

    def __init__(self, index_path: str, movie_ids: np.ndarray):
        self.index = faiss.read_index(index_path)
        self.movie_ids = movie_ids  # must be in the same row order the index was built with

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        exclude_ids: Optional[set] = None,
        overfetch_factor: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the top-k candidates, excluding already-watched movies.

        Since watched movies (already in the user's history/holdout) would
        otherwise eat into the candidate budget -- a heavy watcher could end
        up with only 70-80 *new* candidates out of 100 -- we over-fetch
        (k * overfetch_factor, floor 150) from FAISS first, drop excluded
        ids, then trim back down to exactly k.
        """
        exclude_ids = exclude_ids or set()
        fetch_k = max(int(k * overfetch_factor), k + 50)

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, fetch_k)
        indices = indices[0]
        scores = scores[0]

        candidate_ids = np.full(k, -1, dtype=np.int64)
        candidate_scores = np.zeros(k, dtype=np.float32)
        n_filled = 0
        for idx, score in zip(indices, scores):
            if n_filled >= k:
                break
            if idx < 0:
                continue
            mid = int(self.movie_ids[idx])
            if mid in exclude_ids:
                continue
            candidate_ids[n_filled] = mid
            candidate_scores[n_filled] = score
            n_filled += 1

        return candidate_ids, candidate_scores


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, safe against zero vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

class MovieRecommendationEnv(gym.Env):
    """
    One episode = one user, exactly `episode_length` (default 5) sequential
    recommendations chosen from a fixed set of NUM_CANDIDATES (100) FAISS
    candidates retrieved once at reset() time.

    This is a sequential LIST-CONSTRUCTION task, not a contextual bandit:
    the action at step 3 depends on what was picked at steps 1 and 2, via
    the recommended_mask and running_average_embedding carried in the
    observation.

    Observation (Box(6628,)):
        [ user_embedding(64)
          | candidate_embeddings(100*64)
          | recommended_mask(100)
          | running_average_embedding(64) ]

    Action (Discrete(100)):
        index into the *candidate list for this episode* -- NOT a raw movieId.

    Reward (see _compute_reward):
        On a repeat pick:  REPEAT_PENALTY  (terminated stays False)
        Otherwise:
            reward = W_RELEVANCE  * cosine(user_embedding, selected_embedding)
                    - W_REDUNDANCY * max_cosine_similarity(selected, previous_picks)
                    + W_HOLDOUT    * holdout_hit

    The environment is intentionally modular: additional reward terms
    (genre diversity, tag diversity, popularity penalties, etc.) can be
    added inside _compute_reward without touching reset()/step().
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        movie_embeddings_path: str = "trained_movie_encoder/movie_embeddings.npy",
        movie_ids_path: str = "trained_movie_encoder/movie_ids.npy",
        user_transformer_path: str = "trained_movie_encoder/user_transformer.pth",
        faiss_index_path: str = "trained_movie_encoder/movie_index.faiss",
        ratings_path: str = "ratings.csv",
        num_candidates: int = NUM_CANDIDATES,
        episode_length: int = EPISODE_LENGTH,
        reward_weights: Tuple[float, float, float] = (W_RELEVANCE, W_REDUNDANCY, W_HOLDOUT),
        repeat_penalty: float = REPEAT_PENALTY,
        device: str = "cpu",
        transformer_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
        EMA_ALPHA: float = 0.7
    ):
        super().__init__()

        self.total_candidates = num_candidates

        self.safe_recommendations = SAFE_RECOMMENDATIONS

        self.num_candidates = (
            num_candidates
            - self.safe_recommendations
        )
        self.episode_length = episode_length
        self.w_relevance, self.w_redundancy, self.w_holdout = reward_weights
        self.repeat_penalty = repeat_penalty
        self.device = torch.device(device)
        self.verbose = verbose
        self.ema_alpha = EMA_ALPHA

        # --- data stores ---
        self.movie_store = MovieEmbeddingStore(movie_embeddings_path, movie_ids_path)
        self.history_store = UserHistoryStore(ratings_path, self.movie_store)
        self.retriever = FaissRetriever(faiss_index_path, self.movie_store.movie_ids)

        # --- transformer user encoder ---
        transformer_kwargs = transformer_kwargs or {}
        self.user_encoder = TransformerUserEncoder(embed_dim=EMBED_DIM, **transformer_kwargs)
        state_dict = torch.load(user_transformer_path, map_location=self.device)
        self.user_encoder.load_state_dict(state_dict)
        self.user_encoder.to(self.device)
        self.user_encoder.eval()

        # --- gymnasium spaces ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_candidates)

        # --- rng ---
        self._py_rng = random.Random(seed)

        # --- episode state (populated in reset) ---
        self.current_user_id: Optional[int] = None
        self.greedy_ids = None
        self.greedy_embeddings = None
        self.user_embedding: Optional[np.ndarray] = None
        self.candidate_ids: Optional[np.ndarray] = None
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.recommended_mask: Optional[np.ndarray] = None
        self.selected_embeddings: List[np.ndarray] = []       # history of picks this episode
        self.running_average_embedding: Optional[np.ndarray] = None  # mean(selected_embeddings)
        self.holdout_set: set = set()
        self.step_count: int = 0

    # ------------------------------------------------------------------ #
    # Core Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._py_rng.seed(seed)

        forced_user = None
        if options is not None:
            forced_user = options.get("user_id")

        if forced_user is None:
            user_id = self._py_rng.choice(self.history_store.user_ids)
        else:
            user_id = forced_user
        context_ids, holdout_ids = self.history_store.get_context_and_holdout(user_id)

        user_embedding = self._encode_user(context_ids)

        watched_ids = set(int(m) for m in context_ids) | set(int(m) for m in holdout_ids)
        candidate_ids, _scores = self.retriever.search(
            user_embedding,
            self.total_candidates,
            exclude_ids=watched_ids
        )

        candidate_embeddings, valid_mask = self._safe_candidate_embeddings(
            candidate_ids
        )
        # -------------------------------------------------------
        # Split candidates
        #
        # Top-5:
        #   Greedy recommendations
        #
        # Remaining:
        #   PPO candidate pool
        # -------------------------------------------------------

        self.greedy_ids = candidate_ids[:self.safe_recommendations]

        self.greedy_embeddings = candidate_embeddings[
            :self.safe_recommendations
        ]

        candidate_ids = candidate_ids[
            self.safe_recommendations:
        ]

        candidate_embeddings = candidate_embeddings[
            self.safe_recommendations:
        ]

        valid_mask = valid_mask[
            self.safe_recommendations:
        ]
        self.current_user_id = user_id
        self.user_embedding = user_embedding
        self.candidate_ids = candidate_ids
        self.candidate_embeddings = candidate_embeddings
        # Pre-block any invalid/padded candidate slots (e.g. FAISS returned
        # fewer than num_candidates results) by marking them "already
        # recommended" so the agent effectively cannot select them.
        self.recommended_mask = np.where(valid_mask, 0.0, 1.0).astype(np.float32)
        # -------------------------------------------------------
        # PPO starts AFTER the greedy recommendations.
        #
        # Pretend the greedy movies have already been selected.
        # -------------------------------------------------------

        self.selected_embeddings = [

            emb.copy()

            for emb in self.greedy_embeddings

        ]

        running = self.greedy_embeddings[0].copy()

        for emb in self.greedy_embeddings[1:]:

            running = (

                self.ema_alpha * emb

                +

                (1.0 - self.ema_alpha)

                * running

            )

        running /= (

            np.linalg.norm(running)

            +

            1e-8

        )

        self.running_average_embedding = running.astype(
            np.float32
        )
        self.holdout_set = set(int(m) for m in holdout_ids)
        self.step_count = 0

        if self.verbose:
            n_valid = int(valid_mask.sum())
            print(f"[reset] user_id={user_id} valid_candidates={n_valid}/{self.num_candidates}")

        obs = self._get_observation()
        info = {

            "user_id": user_id,

            "history": list(context_ids),

            "holdout": list(self.holdout_set),

            "greedy_ids": self.greedy_ids.copy(),

            "candidate_ids": self.candidate_ids.copy(),

        }
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        assert self.candidate_ids is not None, "Call reset() before step()."

        movie_id = int(self.candidate_ids[action])
        already_recommended = bool(self.recommended_mask[action] == 1.0)

        reward, reward_terms = self._compute_reward(action, movie_id, already_recommended)

        if not already_recommended:

            movie_embedding = self.candidate_embeddings[action]

            self.recommended_mask[action] = 1.0

            self.selected_embeddings.append(movie_embedding)

            # ----------------------------------------
            # Update running embedding (EMA)
            # ----------------------------------------

            self.running_average_embedding = (

                self.ema_alpha * movie_embedding

                +

                (1.0 - self.ema_alpha)

                * self.running_average_embedding

            ).astype(np.float32)

            self.running_average_embedding /= (

                np.linalg.norm(

                    self.running_average_embedding

                )

                +

                1e-8

            )

        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        # Repeats never end the episode early -- per spec, terminate stays
        # False on a repeat pick; only running out of steps terminates.
        truncated = False

        obs = self._get_observation()
        info = {
            "movie_id": movie_id,
            "already_recommended": already_recommended,
            "step_count": self.step_count,
            **reward_terms,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):  # pragma: no cover - no visualization needed
        pass

    def close(self):  # pragma: no cover
        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _encode_user(self, context_movie_ids: List[int]) -> np.ndarray:
        if len(context_movie_ids) == 0:
            # Degenerate case: no context at all. Feed a single zero vector
            # so the transformer still runs; CLS token dominates.
            history_embeddings = np.zeros((1, EMBED_DIM), dtype=np.float32)
        else:
            history_embeddings = self.movie_store.get_embeddings(context_movie_ids)

        history_tensor = torch.from_numpy(history_embeddings).unsqueeze(0).to(self.device)
        with torch.no_grad():
            user_embedding = self.user_encoder(history_tensor)
        return user_embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def _safe_candidate_embeddings(
        self, candidate_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Look up embeddings for candidate ids, padding any invalid (-1) slots
        with zero vectors so array shapes stay fixed at num_candidates."""
        embeddings = np.zeros(

            (len(candidate_ids), EMBED_DIM),

            dtype=np.float32

        )

        valid_mask = np.zeros(

            len(candidate_ids),

            dtype=bool

        )
        for i, mid in enumerate(candidate_ids):
            if mid >= 0 and self.movie_store.has(mid):
                embeddings[i] = self.movie_store.get_embedding(mid)
                valid_mask[i] = True
        return embeddings, valid_mask

    

    def _compute_reward(
        self, action: int, movie_id: int, already_recommended: bool
    ) -> Tuple[float, dict]:
        """Reward for picking `action` (== index into this episode's
        candidate list), given whether it's a repeat.

            repeat:       reward = REPEAT_PENALTY
            otherwise:    reward = w_relevance  * relevance
                                  - w_redundancy * redundancy
                                  + w_holdout    * holdout_hit

        relevance  = cosine(user_embedding, selected_movie_embedding)
        redundancy = max cosine similarity to every previously selected
                     movie this episode (0.0 if this is the first pick)
        holdout_hit = 1.0 if movie_id is one of the user's held-out
                     future positives, else 0.0

        Kept as its own method (rather than inlined in step()) so new
        reward terms -- genre diversity, tag diversity, popularity
        penalties, etc. -- can be added here without touching reset()/step().
        """
        if already_recommended:
            return self.repeat_penalty, {
                "relevance": 0.0,
                "redundancy": 0.0,
                "holdout_hit": 0.0,
            }

        movie_embedding = self.candidate_embeddings[action]

        relevance = _cosine(self.user_embedding, movie_embedding)

        if self.selected_embeddings:

            redundancy = _cosine(

                movie_embedding,

                self.running_average_embedding

            )

        else:

            redundancy = 0.0
                
        holdout_hit = 1.0 if movie_id in self.holdout_set else 0.0

        reward = (
            self.w_relevance * relevance
            - self.w_redundancy * redundancy
            + self.w_holdout * holdout_hit
        )

        reward_terms = {
            "relevance": relevance,
            "redundancy": redundancy,
            "holdout_hit": holdout_hit,
        }
        return float(reward), reward_terms

    def _get_observation(self) -> np.ndarray:
        obs = np.concatenate(
            [
                self.user_embedding,
                self.candidate_embeddings.flatten(),
                self.recommended_mask,
                self.running_average_embedding,
            ]
        ).astype(np.float32)
        return obs


__all__ = ["MovieRecommendationEnv", "TransformerUserEncoder"]