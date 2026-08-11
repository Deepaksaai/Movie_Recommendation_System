# Adaptive Movie Recommendation System

An end-to-end movie recommendation system that combines representation learning, sequential user modeling, approximate nearest-neighbor retrieval, and reinforcement learning to generate relevant and diverse movie recommendations.

The system uses MovieLens interaction data and Tag Genome movie features to learn movie and user representations, retrieves candidate movies using FAISS, and uses a PPO agent to re-rank recommendations while encouraging relevance and diversity.

---

## Overview

Traditional recommendation systems often optimize immediate relevance, such as predicting whether a user will like a movie.

This project approaches recommendation as a two-stage problem:

1. **Candidate Retrieval** — efficiently retrieve movies relevant to the user's preferences.
2. **Re-ranking** — use reinforcement learning to select recommendations sequentially while considering the interactions between recommended movies.

The main pipeline is:

```text
MovieLens + Tag Genome
          │
          ▼
   Movie Representation
   1084-D Tag Genome
          │
          ▼
      Autoencoder
          │
          ▼
       64-D Movie
       Embeddings
          │
          ▼
    Triplet Fine-tuning
          │
          ▼
   Preference-aware
   Movie Embeddings
          │
          ├──────────────────────┐
          ▼                      ▼
   User Interaction          FAISS Index
       History                   │
          │                      │
          ▼                      │
   Transformer User              │
       Encoder                  │
          │                      │
          ▼                      │
    64-D User Embedding          │
          │                      │
          └──────────┬───────────┘
                     ▼
              Candidate Retrieval
                     │
                     ▼
              Candidate Movies
                     │
                     ▼
                PPO Agent
                     │
                     ▼
             Re-ranked Movies
                     │
                     ▼
               Streamlit UI