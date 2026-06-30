import os
import pickle
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH    = os.environ.get("DATA_PATH", "tagdl.csv")

SAVE_DIR     = "trained_movie_encoder"
LATENT_DIM   = 64
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 3e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED  = 42

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP     = torch.cuda.is_available()
PIN_MEMORY  = torch.cuda.is_available()

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TRIPLET_PATH = "triplets.csv"

TRIPLET_WEIGHT = 0.2

# ============================================================
# LOAD & VALIDATE DATA
# ============================================================

print("=" * 60)
print("Loading Tag Genome...")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\n"
        f"Set the DATA_PATH environment variable or place 'tagdl.csv' "
        f"in the working directory."
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"item_id", "tag", "score"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

print(f"Raw data shape: {df.shape}")

# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

print("\nBuilding Movie x Tag matrix...")

features = (
    df.pivot(index="item_id", columns="tag", values="score")
    .sort_index()
    .fillna(0.0)
)

movie_ids = features.index.to_numpy()
INPUT_DIM = features.shape[1]

# ============================================================
# LOAD TRIPLETS
# ============================================================

triplets = pd.read_csv(TRIPLET_PATH)

valid_movies = set(movie_ids)

triplets = triplets[
    triplets.anchor.isin(valid_movies)
    &
    triplets.positive.isin(valid_movies)
    &
    triplets.negative.isin(valid_movies)
].reset_index(drop=True)

print(f"Valid Triplets : {len(triplets)}")

print(f"Feature matrix: {features.shape}")

assert not np.isnan(features.values).any(), "NaN values found after fillna!"
assert not np.isinf(features.values).any(), "Inf values found in features!"

# ============================================================
# NORMALIZE & SPLIT
# ============================================================

minmax_scaler = MinMaxScaler()
X = minmax_scaler.fit_transform(features.values).astype(np.float32)

# ============================================================
# MOVIE FEATURE LOOKUP
# ============================================================

movie_to_feature = {}

for idx, movie_id in enumerate(movie_ids):

    movie_to_feature[movie_id] = X[idx]

with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(minmax_scaler, f)
with open(os.path.join(SAVE_DIR, "movie_ids.pkl"), "wb") as f:
    pickle.dump(movie_ids, f)

X_train, X_val = train_test_split(X, test_size=0.10, random_state=RANDOM_SEED, shuffle=True)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ============================================================
# DATASET & DATALOADERS
# ============================================================

class MovieDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
class TripletDataset(Dataset):

    def __init__(self, triplets, lookup):

        self.triplets = triplets

        self.lookup = lookup

    def __len__(self):

        return len(self.triplets)

    def __getitem__(self, idx):

        row = self.triplets.iloc[idx]

        anchor = self.lookup[row.anchor]

        positive = self.lookup[row.positive]

        negative = self.lookup[row.negative]

        return (

            torch.tensor(anchor, dtype=torch.float32),

            torch.tensor(positive, dtype=torch.float32),

            torch.tensor(negative, dtype=torch.float32)

        )

train_loader = DataLoader(MovieDataset(X_train), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=PIN_MEMORY)
val_loader   = DataLoader(MovieDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

triplet_loader = DataLoader(

    TripletDataset(

        triplets,

        movie_to_feature

    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    pin_memory=PIN_MEMORY

)

print("\nDataset Ready.")
print("=" * 60)

# ============================================================
# MODEL
# ============================================================

class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class Block(nn.Module):
    """Linear -> LayerNorm -> GELU with residual connection."""
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class MovieAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            Block(input_dim, 512),
            Block(512, 256),
            nn.Linear(256, latent_dim),
            L2Normalize()
        )

        self.decoder = nn.Sequential(
            Block(latent_dim, 256),
            Block(256, 512),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


model = MovieAutoEncoder(INPUT_DIM, LATENT_DIM).to(DEVICE)

for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

print(model)

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

criterion = nn.MSELoss()
triplet_criterion = nn.TripletMarginLoss(

    margin=1.0,

    p=2

)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# CosineAnnealingLR smoothly decays LR over all epochs — no premature halving
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

scaler_amp = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float("inf")
train_losses, val_losses = [], []

print("\nStarting Training...")
print("=" * 60)

triplet_iterator = iter(triplet_loader)

for epoch in range(EPOCHS):

    # ── Train ──────────────────────────────────────────────
    model.train()
    running_train = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        try:

            anchor, positive, negative = next(triplet_iterator)

        except StopIteration:

            triplet_iterator = iter(triplet_loader)

            anchor, positive, negative = next(triplet_iterator)

        anchor = anchor.to(DEVICE)

        positive = positive.to(DEVICE)

        negative = negative.to(DEVICE)

        with torch.autocast(device_type=DEVICE_TYPE, enabled=USE_AMP):

            reconstruction, _ = model(batch)

            reconstruction_loss = criterion(

                reconstruction,

                batch

            )

            anchor_embedding = model.encoder(anchor)

            positive_embedding = model.encoder(positive)

            negative_embedding = model.encoder(negative)

            metric_loss = triplet_criterion(

                anchor_embedding,

                positive_embedding,

                negative_embedding

            )

            loss = reconstruction_loss + TRIPLET_WEIGHT * metric_loss

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at epoch {epoch+1}, skipping batch.")
            continue

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        running_train += loss.item()

        if epoch % 5 == 0:

            print(

                f"Recon: {reconstruction_loss.item():.5f}"

                f" | Triplet: {metric_loss.item():.5f}"

            )

    avg_train = running_train / len(train_loader)

    # ── Validate ───────────────────────────────────────────
    model.eval()
    running_val = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
            batch = batch.to(DEVICE)
            reconstruction, _ = model(batch)
            running_val += criterion(reconstruction, batch).item()

    avg_val = running_val / len(val_loader)

    train_losses.append(avg_train)
    val_losses.append(avg_val)

    scheduler.step()

    # ── Save best ──────────────────────────────────────────
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(),         os.path.join(SAVE_DIR, "best_autoencoder.pth"))
        torch.save(model.encoder.state_dict(), os.path.join(SAVE_DIR, "best_encoder.pth"))
        print(f"\n  + New Best Saved (Val Loss = {best_val_loss:.6f})")

    print(
        f"Epoch [{epoch+1:03d}/{EPOCHS}] | "
        f"Train: {avg_train:.6f} | "
        f"Val: {avg_val:.6f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

print("\nTraining Complete.")
print("=" * 60)

# ============================================================
# LOAD BEST & GENERATE EMBEDDINGS
# ============================================================

print("\nLoading Best Model...")

best_path = os.path.join(SAVE_DIR, "best_autoencoder.pth")
if not os.path.exists(best_path):
    print("WARNING: No best checkpoint found — using current model weights.")
    torch.save(model.state_dict(), best_path)

model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
model.eval()

print("Generating Movie Embeddings...")

with torch.no_grad():
    all_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    embeddings = model.encoder(all_tensor).cpu().numpy()

print(f"Embeddings Shape : {embeddings.shape}")

# ============================================================
# SAVE EVERYTHING
# ============================================================

np.save(os.path.join(SAVE_DIR, "movie_embeddings.npy"), embeddings)
np.save(os.path.join(SAVE_DIR, "movie_ids.npy"), movie_ids)

torch.save(model.state_dict(),         os.path.join(SAVE_DIR, "final_autoencoder.pth"))
torch.save(model.encoder.state_dict(), os.path.join(SAVE_DIR, "movie_encoder.pth"))

with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
    json.dump({
        "latent_dim":    LATENT_DIM,
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LR,
        "input_dim":     INPUT_DIM
    }, f, indent=4)

# ============================================================
# PLOT TRAINING CURVE
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Movie AutoEncoder Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
plt.close()

# ============================================================
# SUMMARY
# ============================================================

print("\nEmbedding Statistics")
print("-" * 40)
print(f"Mean      : {embeddings.mean():.6f}")
print(f"Std       : {embeddings.std():.6f}")
print(f"Min       : {embeddings.min():.6f}")
print(f"Max       : {embeddings.max():.6f}")
print(f"Dimension : {embeddings.shape[1]}")
print(f"Movies    : {embeddings.shape[0]}")
print("-" * 40)
print(f"\nAll artifacts saved to: {SAVE_DIR}")