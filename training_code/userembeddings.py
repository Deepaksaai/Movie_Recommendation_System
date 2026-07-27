"""
Generate User Embeddings
========================
Loads:
    trained_movie_encoder/movie_embeddings.npy
    trained_movie_encoder/movie_ids.npy
    trained_movie_encoder/user_transformer.pth
    ratings.csv

Saves:
    trained_movie_encoder/user_embeddings.npy
    trained_movie_encoder/user_ids.npy
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

EMBEDDINGS_PATH  = "trained_movie_encoder/movie_embeddings.npy"
IDS_PATH         = "trained_movie_encoder/movie_ids.npy"
WEIGHTS_PATH     = "trained_movie_encoder/user_transformer.pth"
RATINGS_PATH     = "ratings.csv"
OUT_EMBEDDINGS   = "trained_movie_encoder/user_embeddings.npy"
OUT_IDS          = "trained_movie_encoder/user_ids.npy"

POS_THRESHOLD    = 4.0
MAX_HISTORY      = 100
MIN_HISTORY      = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MovieHistoryTransformer(nn.Module):
    def __init__(
        self,
        embed_dim:       int   = 64,
        nhead:           int   = 4,
        num_layers:      int   = 2,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_enc   = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection  = nn.Linear(embed_dim, embed_dim)

    def forward(self, history_embs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B        = history_embs.size(0)
        cls      = self.cls_token.expand(B, -1, -1)
        x        = torch.cat([cls, history_embs], dim=1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
        mask     = torch.cat([cls_mask, padding_mask], dim=1)
        x        = self.pos_enc(x)
        x        = self.transformer(x, src_key_padding_mask=mask)
        user_emb = self.projection(x[:, 0, :])
        return nn.functional.normalize(user_emb, p=2, dim=1)


# ─────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────

def encode_user(
    model:            MovieHistoryTransformer,
    history_ids:      list,
    embedding_matrix: torch.Tensor,
    movie_id_to_idx:  dict,
) -> np.ndarray:
    """Return a (64,) user vector for one user's watch history."""
    embs = torch.stack(
        [embedding_matrix[movie_id_to_idx[mid]] for mid in history_ids]
    ).unsqueeze(0).to(DEVICE)                                    # (1, L, 64)

    mask = torch.zeros(1, len(history_ids), dtype=torch.bool, device=DEVICE)
    return model(embs, mask).squeeze(0).cpu().numpy()            # (64,)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # ── Load movie embeddings ──────────────────────────────────
    print("Loading movie embeddings...")
    raw_embeddings  = np.load(EMBEDDINGS_PATH)                   # (N, 64)
    movie_ids_array = np.load(IDS_PATH, allow_pickle=True)
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids_array)}
    embedding_matrix = torch.tensor(raw_embeddings, dtype=torch.float32).to(DEVICE)
    print(f"  Movies with embeddings: {len(movie_id_to_idx):,}\n")

    # ── Load model ────────────────────────────────────────────
    print("Loading transformer weights...")
    model = MovieHistoryTransformer().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print("  Weights loaded.\n")

    # ── Load & filter ratings ─────────────────────────────────
    print("Loading ratings...")
    ratings = pd.read_csv(
        RATINGS_PATH,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )
    print(f"  Total ratings:          {len(ratings):,}")

    ratings = ratings[ratings["movieId"].isin(movie_id_to_idx)]
    print(f"  After embedding filter: {len(ratings):,}")

    ratings = ratings[ratings["rating"] >= POS_THRESHOLD]
    print(f"  After rating filter:    {len(ratings):,}")

    ratings = ratings.sort_values(["userId", "timestamp"])
    print()

    # ── Generate embeddings ───────────────────────────────────
    print("Generating user embeddings...")
    user_ids_out   = []
    user_embs_out  = []

    with torch.no_grad():
        for user_id, group in tqdm(ratings.groupby("userId")):
            history = group["movieId"].tolist()

            if len(history) < MIN_HISTORY:
                continue

            history  = history[-MAX_HISTORY:]
            user_vec = encode_user(model, history, embedding_matrix, movie_id_to_idx)

            user_ids_out.append(user_id)
            user_embs_out.append(user_vec)

    # ── Save ──────────────────────────────────────────────────
    user_ids_arr  = np.array(user_ids_out)
    user_embs_arr = np.stack(user_embs_out)

    np.save(OUT_IDS,        user_ids_arr)
    np.save(OUT_EMBEDDINGS, user_embs_arr)

    print()
    print("=" * 45)
    print("Done!")
    print(f"  Users saved:       {len(user_ids_arr):,}")
    print(f"  Embedding shape:   {user_embs_arr.shape}")
    print(f"  Saved to:          {OUT_IDS}")
    print(f"                     {OUT_EMBEDDINGS}")
    print("=" * 45)


if __name__ == "__main__":
    main()