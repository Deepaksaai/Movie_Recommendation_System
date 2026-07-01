"""
Build FAISS Index
=================

Loads
-----
trained_movie_encoder/movie_embeddings.npy
trained_movie_encoder/movie_ids.npy

Saves
-----
trained_movie_encoder/movie_index.faiss

Uses cosine similarity via IndexFlatIP.
Movie embeddings MUST already be L2 normalized.
"""

import os
import faiss
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

EMBEDDINGS_PATH = "trained_movie_encoder/movie_embeddings.npy"
MOVIE_IDS_PATH  = "trained_movie_encoder/movie_ids.npy"

OUTPUT_INDEX = "trained_movie_encoder/movie_index.faiss"

# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================

print("Loading movie embeddings...")

embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
movie_ids  = np.load(MOVIE_IDS_PATH)

print(f"Movies      : {len(movie_ids):,}")
print(f"Dimensions  : {embeddings.shape[1]}")

# ==========================================================
# VERIFY NORMALIZATION
# ==========================================================

norms = np.linalg.norm(embeddings, axis=1)

print()
print("Embedding Statistics")
print("----------------------------")
print(f"Mean L2 Norm : {norms.mean():.6f}")
print(f"Min  L2 Norm : {norms.min():.6f}")
print(f"Max  L2 Norm : {norms.max():.6f}")

if not np.allclose(norms, 1.0, atol=1e-3):
    print("\nEmbeddings are not perfectly normalized.")
    print("Normalizing before indexing...")

    faiss.normalize_L2(embeddings)

# ==========================================================
# BUILD INDEX
# ==========================================================

print("\nBuilding FAISS Index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# ==========================================================
# SAVE
# ==========================================================

faiss.write_index(index, OUTPUT_INDEX)

print("\n========================================")
print("FAISS index created successfully!")
print(f"Index Type   : IndexFlatIP")
print(f"Movies       : {index.ntotal:,}")
print(f"Dimensions   : {dimension}")
print(f"Saved Index  : {OUTPUT_INDEX}")
print("========================================")