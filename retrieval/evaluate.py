"""
evaluate.py
===========

Offline evaluation of the retrieval stage.

Pipeline:

History
    ↓
Transformer
    ↓
User Embedding
    ↓
FAISS
    ↓
Top-K Retrieval
    ↓
Recall@K / HitRate@K / NDCG@K
"""

import math
import faiss
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_PATH = "trained_movie_encoder/movie_embeddings.npy"
MOVIE_IDS_PATH  = "trained_movie_encoder/movie_ids.npy"

TRANSFORMER_PATH = "trained_movie_encoder/user_transformer.pth"

INDEX_PATH = "trained_movie_encoder/movie_index.faiss"

RATINGS_PATH = "ratings.csv"

TOP_K = [10, 50, 100]

POS_THRESHOLD = 4.0

MIN_HISTORY = 3

MAX_HISTORY = 100

# ==========================================================
# POSITIONAL ENCODING
# ==========================================================


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512,
        dropout=0.1
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(

            torch.arange(0, d_model, 2).float()

            * (-math.log(10000.0) / d_model)

        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(

            "pe",

            pe.unsqueeze(0)

        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


# ==========================================================
# TRANSFORMER
# ==========================================================


class MovieHistoryTransformer(nn.Module):

    def __init__(

        self,

        embed_dim=64,

        nhead=4,

        num_layers=2,

        dim_feedforward=256,

        dropout=0.1,

    ):

        super().__init__()

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

    def forward(

        self,

        history_embs,

        padding_mask

    ):

        B = history_embs.size(0)

        cls = self.cls_token.expand(

            B,

            -1,

            -1

        )

        x = torch.cat(

            [cls, history_embs],

            dim=1

        )

        cls_mask = torch.zeros(

            B,

            1,

            dtype=torch.bool,

            device=padding_mask.device

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

        user_emb = nn.functional.normalize(

            user_emb,

            p=2,

            dim=1

        )

        return user_emb


# ==========================================================
# LOAD EVERYTHING
# ==========================================================

print("Loading movie embeddings...")

movie_embeddings = np.load(

    EMBEDDINGS_PATH

).astype(np.float32)

movie_ids = np.load(

    MOVIE_IDS_PATH

)

movie_id_to_idx = {

    int(mid): i

    for i, mid in enumerate(movie_ids)

}

embedding_matrix = torch.tensor(

    movie_embeddings,

    dtype=torch.float32,

    device=DEVICE

)

print(f"Movies: {len(movie_ids):,}")

print("Loading FAISS index...")

index = faiss.read_index(

    INDEX_PATH

)

print(f"Indexed movies: {index.ntotal:,}")

print("Loading Transformer...")

model = MovieHistoryTransformer().to(

    DEVICE

)

model.load_state_dict(

    torch.load(

        TRANSFORMER_PATH,

        map_location=DEVICE

    )

)

model.eval()

print("Transformer loaded.\n")
# ==========================================================
# USER ENCODING
# ==========================================================

@torch.no_grad()
def encode_user_history(history):

    idxs = [
        movie_id_to_idx[mid]
        for mid in history
        if mid in movie_id_to_idx
    ]

    if len(idxs) < MIN_HISTORY:
        return None

    embs = embedding_matrix[idxs].unsqueeze(0)

    padding_mask = torch.zeros(
        (1, len(idxs)),
        dtype=torch.bool,
        device=DEVICE
    )

    user_emb = model(
        embs,
        padding_mask
    )

    return user_emb.cpu().numpy().astype(np.float32)


# ==========================================================
# METRICS
# ==========================================================

def recall_at_k(retrieved, ground_truth, k):

    retrieved = retrieved[:k]

    hits = len(
        set(retrieved).intersection(
            ground_truth
        )
    )

    return hits / len(ground_truth)


def hit_rate_at_k(retrieved, ground_truth, k):

    retrieved = retrieved[:k]

    return float(

        len(
            set(retrieved).intersection(
                ground_truth
            )
        ) > 0

    )


def ndcg_at_k(retrieved, ground_truth, k):

    retrieved = retrieved[:k]

    dcg = 0.0

    for rank, movie in enumerate(retrieved):

        if movie in ground_truth:

            dcg += 1.0 / np.log2(rank + 2)

    ideal_hits = min(

        len(ground_truth),

        k

    )

    idcg = 0.0

    for i in range(ideal_hits):

        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:

        return 0.0

    return dcg / idcg


# ==========================================================
# LOAD RATINGS
# ==========================================================

print("Loading ratings...")

ratings = pd.read_csv(

    RATINGS_PATH,

    usecols=[
        "userId",
        "movieId",
        "rating",
        "timestamp"
    ]

)

ratings = ratings[

    ratings.movieId.isin(
        movie_id_to_idx.keys()
    )

]

ratings = ratings[

    ratings.rating >= POS_THRESHOLD

]

ratings = ratings.sort_values(

    ["userId", "timestamp"]

)

print(f"Positive Ratings : {len(ratings):,}")

groups = ratings.groupby("userId")

print(f"Users            : {len(groups):,}")

print()


# ==========================================================
# HISTORY SPLIT
# ==========================================================

def split_history(group):

    history = group.movieId.tolist()

    if len(history) < 10:

        return None

    split = int(

        len(history) * 0.8

    )

    train_history = history[:split]

    future_movies = history[split:]

    train_history = train_history[-MAX_HISTORY:]

    if len(train_history) < MIN_HISTORY:

        return None

    return train_history, future_movies
# ==========================================================
# EVALUATION
# ==========================================================

results = {

    "Recall@10": [],

    "Recall@50": [],

    "Recall@100": [],

    "HitRate@10": [],

    "HitRate@50": [],

    "HitRate@100": [],

    "NDCG@10": [],

    "NDCG@50": [],

    "NDCG@100": []

}

evaluated_users = 0

print("Evaluating Retrieval...\n")

for user_id, group in tqdm(groups):

    split = split_history(group)

    if split is None:

        continue

    history, future_movies = split

    user_embedding = encode_user_history(history)

    if user_embedding is None:

        continue

    # --------------------------------------
    # Search FAISS
    # --------------------------------------

    scores, indices = index.search(

        user_embedding,

        200

    )

    history_set = set(history)

    retrieved = []

    for idx in indices[0]:

        movie_id = int(movie_ids[idx])

        if movie_id in history_set:

            continue

        retrieved.append(movie_id)

        if len(retrieved) >= 100:

            break

    if len(retrieved) == 0:

        continue

    future_set = set(future_movies)

    results["Recall@10"].append(

        recall_at_k(

            retrieved,

            future_set,

            10

        )

    )

    results["Recall@50"].append(

        recall_at_k(

            retrieved,

            future_set,

            50

        )

    )

    results["Recall@100"].append(

        recall_at_k(

            retrieved,

            future_set,

            100

        )

    )

    results["HitRate@10"].append(

        hit_rate_at_k(

            retrieved,

            future_set,

            10

        )

    )

    results["HitRate@50"].append(

        hit_rate_at_k(

            retrieved,

            future_set,

            50

        )

    )

    results["HitRate@100"].append(

        hit_rate_at_k(

            retrieved,

            future_set,

            100

        )

    )

    results["NDCG@10"].append(

        ndcg_at_k(

            retrieved,

            future_set,

            10

        )

    )

    results["NDCG@50"].append(

        ndcg_at_k(

            retrieved,

            future_set,

            50

        )

    )

    results["NDCG@100"].append(

        ndcg_at_k(

            retrieved,

            future_set,

            100

        )

    )

    evaluated_users += 1


# ==========================================================
# RESULTS
# ==========================================================

print()

print("=" * 60)

print("Retrieval Evaluation Results")

print("=" * 60)

print(f"Users Evaluated : {evaluated_users:,}")

print()

for metric, values in results.items():

    print(f"{metric:<15}: {np.mean(values):.4f}")

print("=" * 60)