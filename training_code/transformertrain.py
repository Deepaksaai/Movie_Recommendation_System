"""
Transformer-based User Embedding Model for Movie Recommendations
================================================================
Architecture:
  History of movie embeddings (64-D each)
    → Positional Encoding
    → 2x Transformer Encoder Layers
    → CLS Token pooling
    → Linear projection
    → 64-D User Embedding

Trained with TripletMarginLoss against pre-computed movie embeddings.
"""

from matplotlib.pylab import sample
import numpy as np
import pickle
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random




# ─────────────────────────────────────────────
# 1. Load pre-computed movie embeddings
# ─────────────────────────────────────────────

def load_movie_embeddings(embeddings_path: str, ids_path: str):
    """
    Returns
    -------
    embedding_matrix : torch.Tensor  [N, 64]
    movie_id_to_idx  : dict          {movie_id: row_index}
    idx_to_movie_id  : dict          {row_index: movie_id}
    """
    embeddings = np.load(embeddings_path)          # (N, 64)
    movie_ids  = np.load(ids_path, allow_pickle=True)

    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie_id = {i: mid for i, mid in enumerate(movie_ids)}

    embedding_matrix = torch.tensor(embeddings, dtype=torch.float32)
    return embedding_matrix, movie_id_to_idx, idx_to_movie_id


# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────

class UserHistoryDataset(Dataset):
    """
    Each sample is a (history, positive, negative) triplet.

    Parameters
    ----------
    triplets           : list of (history_ids, positive_id, negative_id)
    embedding_matrix   : torch.Tensor [N, 64]
    movie_id_to_idx    : dict
    """

    def __init__(self, triplets, embedding_matrix, movie_id_to_idx):
        self.triplets          = triplets
        self.embedding_matrix  = embedding_matrix
        self.movie_id_to_idx   = movie_id_to_idx

    def _lookup(self, movie_id):

        if movie_id not in self.movie_id_to_idx:
            print(f"Movie ID {movie_id} not found!")
            raise KeyError(movie_id)

        idx = self.movie_id_to_idx[movie_id]
        return self.embedding_matrix[idx]      # (64,)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):

        user = self.triplets[i]

        positives = user["positives"]
        negatives = user["negatives"]

        split = random.randint(2, len(positives) - 1)

        MAX_HISTORY = 100

        start = max(0, split - MAX_HISTORY)

        history_ids = positives[start:split]

        pos_id = positives[split]

        neg_id = random.choice(negatives)

        history_embs = torch.stack(
            [self._lookup(mid) for mid in history_ids]
        )

        pos_emb = self._lookup(pos_id)

        neg_emb = self._lookup(neg_id)

        return history_embs, pos_emb, neg_emb


# ─────────────────────────────────────────────
# 3. Collate function — padding + attention mask
# ─────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads variable-length histories to the longest sequence in the batch
    and builds a boolean padding mask for the Transformer.

    Returns
    -------
    padded_histories : (B, max_len, 64)
    padding_mask     : (B, max_len)   True  → position is PAD (ignored)
    pos_embs         : (B, 64)
    neg_embs         : (B, 64)
    """
    history_embs, pos_embs, neg_embs = zip(*batch)

    lengths = [h.shape[0] for h in history_embs]
    max_len = max(lengths)

    B   = len(history_embs)
    DIM = history_embs[0].shape[1]

    padded_histories = torch.zeros(B, max_len, DIM)
    padding_mask     = torch.ones(B, max_len, dtype=torch.bool)  # True = ignore

    for i, (h, l) in enumerate(zip(history_embs, lengths)):
        padded_histories[i, :l, :] = h
        padding_mask[i, :l]        = False          # real tokens → not masked

    pos_embs = torch.stack(pos_embs)               # (B, 64)
    neg_embs = torch.stack(neg_embs)               # (B, 64)

    return padded_histories, padding_mask, pos_embs, neg_embs


# ─────────────────────────────────────────────
# 4. Positional Encoding
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 5. The Model
# ─────────────────────────────────────────────

class MovieHistoryTransformer(nn.Module):
    """
    Input  : padded history embeddings  (B, seq_len, 64)
    Output : user embedding             (B, 64)

    A learnable CLS token is prepended to the sequence.
    Its output representation is projected to the final user embedding.
    """

    def __init__(
        self,
        embed_dim:   int = 64,
        nhead:       int = 4,
        num_layers:  int = 2,
        dim_feedforward: int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,   # (B, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, history_embs, padding_mask):
        """
        Parameters
        ----------
        history_embs : (B, seq_len, 64)
        padding_mask : (B, seq_len)   True = PAD position

        Returns
        -------
        user_emb : (B, 64)
        """
        B = history_embs.size(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, 64)
        x   = torch.cat([cls, history_embs], dim=1)           # (B, 1+seq_len, 64)

        # Extend mask: CLS is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
        mask     = torch.cat([cls_mask, padding_mask], dim=1) # (B, 1+seq_len)

        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=mask)    # (B, 1+seq_len, 64)

        cls_out  = x[:, 0, :]                                 # (B, 64)
        user_emb = self.projection(cls_out)

        user_emb = nn.functional.normalize(
            user_emb,
            p=2,
            dim=1
        )

        return user_emb

# ─────────────────────────────────────────────
# 6. Training loop
# ─────────────────────────────────────────────

def train(
    triplets_path:    str   = "triplets.pkl",
    embeddings_path:  str   = "trained_movie_encoder/movie_embeddings.npy",
    ids_path:         str   = "trained_movie_encoder/movie_ids.npy",
    save_path:        str   = "trained_movie_encoder/user_transformer.pth",
    epochs:           int   = 20,
    batch_size:       int   = 64,
    lr:               float = 1e-4,
    margin:           float = 0.5,
    device:           str   = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # ── Data ──────────────────────────────────
    embedding_matrix, movie_id_to_idx, _ = load_movie_embeddings(
        embeddings_path, ids_path
    )
    embedding_matrix = embedding_matrix.to(device)

    with open(triplets_path, "rb") as f:
        users = pickle.load(f)

    print(f"Users: {len(users)}")
    print(users[0])

    dataset = UserHistoryDataset(
        users,
        embedding_matrix.cpu(),
        movie_id_to_idx
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────
    model = MovieHistoryTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.TripletMarginLoss(margin=margin, p=2)

    # ── Loop ──────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for history_embs, padding_mask, pos_embs, neg_embs in dataloader:
            history_embs = history_embs.to(device)
            padding_mask = padding_mask.to(device)
            pos_embs     = pos_embs.to(device)
            neg_embs     = neg_embs.to(device)

            user_embs = model(history_embs, padding_mask)   # (B, 64)

            loss = loss_fn(user_embs, pos_embs, neg_embs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:03d}/{epochs}  |  Loss: {avg_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → {save_path}")
    return model


# ─────────────────────────────────────────────
# 7. Inference helper
# ─────────────────────────────────────────────

def encode_user_history(model, history_ids, embedding_matrix, movie_id_to_idx, device="cpu"):
    """
    Convert a single user's watch history into a 64-D user vector.

    Parameters
    ----------
    model           : trained MovieHistoryTransformer
    history_ids     : list of movie IDs (ints)
    embedding_matrix: torch.Tensor [N, 64]
    movie_id_to_idx : dict

    Returns
    -------
    user_vector : np.ndarray (64,)
    """
    model.eval()
    with torch.no_grad():
        embs = torch.stack([
            embedding_matrix[movie_id_to_idx[mid]] for mid in history_ids
        ]).unsqueeze(0).to(device)                            # (1, seq_len, 64)

        mask = torch.zeros(1, len(history_ids), dtype=torch.bool, device=device)
        user_vec = model(embs, mask).squeeze(0).cpu().numpy()

    return user_vec                                           # (64,)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train(
        triplets_path   = "transformer_users.pkl",
        embeddings_path = "trained_movie_encoder/movie_embeddings.npy",
        ids_path        = "trained_movie_encoder/movie_ids.npy",
        save_path       = "trained_movie_encoder/user_transformer.pth",
        epochs          = 20,
        batch_size      = 64,
        lr              = 1e-4,
        margin          = 0.5,
    )