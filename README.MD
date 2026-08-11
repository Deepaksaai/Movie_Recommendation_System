# Adaptive Movie Recommendation System

An end-to-end movie recommendation system that combines representation learning, sequential user modeling, approximate nearest-neighbor retrieval, and reinforcement learning to generate relevant and diverse movie recommendations.

The system uses MovieLens interaction data and Tag Genome movie features to learn movie and user representations, retrieves candidate movies using FAISS, and uses a PPO agent to re-rank recommendations while encouraging relevance and diversity.

---

## Overview

Traditional recommendation systems often optimize immediate relevance, such as predicting whether a user will like a movie.

This project approaches recommendation as a two-stage problem:

1. **Candidate Retrieval** — efficiently retrieve movies relevant to the user's preferences.
2. **Re-ranking** — use reinforcement learning to select recommendations sequentially while considering the interactions between recommended movies.

The main pipeline:

```
MovieLens + Tag Genome
        │
        ▼
Movie Representation (1084-D Tag Genome)
        │
        ▼
    Autoencoder
        │
        ▼
  64-D Movie Embeddings
        │
        ▼
  Triplet Fine-tuning
        │
        ▼
Preference-aware Movie Embeddings
        │
        ├─────────────────────────────┐
        ▼                             ▼
User Interaction History         FAISS Index
        │                             │
        ▼                             │
Transformer User Encoder              │
        │                             │
        ▼                             │
  64-D User Embedding                 │
        │                             │
        └─────────────┬───────────────┘
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
```

## Features

- 1084-dimensional Tag Genome movie representations
- Autoencoder-based dimensionality reduction to 64 dimensions
- Triplet-loss fine-tuning using MovieLens user interactions
- Transformer-based sequential user embeddings
- FAISS approximate nearest-neighbor retrieval
- PPO-based recommendation re-ranking
- Relevance and diversity-aware recommendation environment
- Top-rated movie history visualization
- TMDB movie posters and metadata
- Interactive Streamlit application
- Existing-user recommendation demo
- Movie-selection interface for building a new user profile

---

## 1. Movie Representation Learning

### Tag Genome

The Tag Genome dataset provides relevance scores for 1,084 tags associated with movies.

Instead of representing a movie only by its ID or a small set of genres, these features provide a richer representation of movie characteristics.

Each movie initially has:

```
Movie → 1084-dimensional Tag Genome vector
```

### Autoencoder

An autoencoder compresses the 1084-dimensional Tag Genome representation into a compact 64-dimensional latent space.

```
1084-D Input → Encoder → 64-D → Decoder → 1084-D Reconstruction
```

The resulting 64-dimensional latent representation becomes the initial movie embedding. This provides:

- dimensionality reduction
- a compact representation
- faster similarity search
- reduced redundancy

---

## 2. Preference-aware Movie Embeddings

Reconstruction alone does not explicitly make the embedding space useful for recommendation.

The movie embeddings were therefore further fine-tuned using MovieLens user interaction data.

Triplets were constructed from user ratings:

```
                Anchor Movie
                     │
             ┌───────┴───────┐
             ▼               ▼
      Positive Movie   Negative Movie
       (User liked)    (User disliked)
```

The objective is to make the anchor closer to the positive movie than to the negative movie:

```
distance(anchor, positive) < distance(anchor, negative)
```

This moves the embedding space from simply representing Tag Genome similarity toward representing user preference similarity.

The final movie representation is a **64-dimensional preference-aware embedding**.

---

## 3. User Embeddings

The movie embeddings are then used to represent users.

A user's interaction history is converted from movie IDs into a sequence of 64-dimensional movie embeddings.

```
User History → Movie IDs → 64-D Movie Embeddings → Transformer Encoder
             → CLS Representation → 64-D User Embedding
```

The Transformer processes the sequence of movies in the user's history and produces a fixed-length representation of the user's preferences.

Training samples are constructed chronologically from MovieLens interactions. A typical training sample contains:

```
History → Positive Movie → Negative Movie
```

The model learns to produce a user representation that is closer to movies the user is likely to prefer. The result is a user embedding in the same 64-dimensional space as the movie embeddings.

---

## 4. FAISS Candidate Retrieval

The system uses FAISS for efficient nearest-neighbor retrieval.

Searching the entire movie catalog for every recommendation would become expensive as the catalog grows. Instead:

```
64-D User Embedding → FAISS → Nearest Movie Embeddings → Candidate Movies
```

FAISS retrieves the movies closest to the user's embedding. These candidates are then passed to the recommendation stage.

---

## 5. PPO Re-ranking

FAISS provides highly relevant candidates, but the most similar movies can also be highly repetitive.

The recommendation problem is therefore treated as a sequential decision problem. The PPO environment maintains information about:

- the user's preference representation
- candidate movies
- movies already recommended
- similarity between selected movies
- recommendation diversity
- duplicate selections

The PPO agent selects movies sequentially. The reward is designed to encourage useful recommendations while discouraging excessive similarity and repeated recommendations.

```
FAISS Candidates → PPO Agent → Sequential Selection → Final Recommendations
```

PPO was trained offline using Stable-Baselines3.

---

## 6. Final Recommendation Strategy

The final recommendation interface combines retrieval-based recommendations with PPO re-ranking.

The current presentation consists of:

```
Top 5 Greedy Recommendations + Top 5 PPO Recommendations
```

The PPO recommendation search space excludes movies already selected by the greedy stage. This allows the final recommendation list to combine highly relevant retrieved movies with additional recommendations selected by the RL policy.

---

## 7. Evaluation

The project includes evaluation using recommendation metrics such as:

- Recall@K
- NDCG@K
- Hit Rate
- Coverage
- Diversity
- Average pairwise similarity

The recommendation environment also uses held-out interactions to evaluate whether recommended movies correspond to movies the user interacted with later in their sequence.

---

## 8. Streamlit Application

A Streamlit application provides an interactive interface for the recommendation system.

### Existing User Demo

Users can select a pre-existing MovieLens user for whom a learned user embedding is available.

The application displays:

```
Top Rated Movies → Greedy Recommendations → PPO Recommendations
```

The top-rated movies show the user's highest-rated movies and their ratings, providing an interpretable view of the user's preferences.

Movie cards include:

- poster
- title
- year
- genres
- rating (where applicable)

TMDB is used to retrieve movie posters and presentation metadata.

### Build Your Own User

The application also provides a movie-selection interface where users can search for movies and construct their own movie history.

The intended pipeline is:

```
Selected Movies → User Representation → FAISS Retrieval → PPO Re-ranking → Recommendations
```

The movie-selection UI is implemented. The full cold-start backend path for generating recommendations directly from an arbitrary list of selected movies is the next extension of the system.

---

## 9. Project Structure

```
.
├── backend/
│   └── recommender.py
│
├── retrieval/
│   ├── build_faiss.py
│   ├── retrieve.py
│   └── evaluate.py
│
├── rl/
│   ├── environment.py
│   ├── train_ppo.py
│   ├── train_a2c.py
│   ├── train_dqn.py
│   └── ppo_inference.py
│
├── training_code/
│   ├── train_movie_encoder.py
│   ├── transformerdata.py
│   ├── transformertrain.py
│   ├── triplets.py
│   └── userembeddings.py
│
├── movie_rec_app/
│   └── app/
│       ├── app.py
│       ├── views/
│       │   ├── existing_user.py
│       │   └── build_user.py
│       ├── components/
│       │   ├── movie_card.py
│       │   ├── recommendation_grid.py
│       │   └── search_box.py
│       └── utils/
│           ├── backend.py
│           ├── data.py
│           └── tmdb.py
│
├── trained_models/
├── trained_movie_encoder/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 10. Technologies

**Machine Learning**
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

**Representation Learning**
- Autoencoders
- Triplet Loss
- Transformer Encoder

**Information Retrieval**
- FAISS
- Cosine Similarity
- Approximate Nearest-Neighbor Search

**Reinforcement Learning**
- PPO
- Stable-Baselines3
- Gymnasium

**Application**
- Streamlit
- TMDB API

---

## 11. Datasets

**MovieLens**

MovieLens provides the user interaction data used for:

- user histories
- ratings
- timestamps
- sequential training samples
- preference-aware embedding training
- evaluation

**Tag Genome**

Tag Genome provides 1,084-dimensional tag relevance features used to learn the initial movie representations.

**TMDB**

TMDB is used by the application layer for movie posters and additional presentation metadata.

---

## 12. Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The trained model files and required datasets must be available locally.

If using TMDB posters, configure the TMDB API key as an environment variable, for example:

```bash
TMDB_API_KEY=your_api_key
```



---

## 13. Running the Application

From the project root:

```bash
streamlit run movie_rec_app/app/app.py
```

The Streamlit application will open in the browser.

---

## 14. Recommendation Flow

For an existing user:

```
MovieLens User
      │
      ▼
Interaction History
      │
      ▼
Transformer User Encoder
      │
      ▼
64-D User Embedding
      │
      ▼
FAISS Candidate Retrieval
      │
      ├─────────────────┐
      ▼                 ▼
  Top 5 Greedy       PPO Re-ranking
                          │
                          ▼
                      Top 5 PPO
      │                   │
      └─────────┬─────────┘
                ▼
         Final 10 Movies
```

---

## 15. Future Improvements

Potential improvements include:

- Complete cold-start recommendations from arbitrary user-selected movies
- Online user feedback and policy updates
- Richer movie metadata such as cast, directors, and plot descriptions
- Larger-scale interaction datasets
- More sophisticated diversity objectives
- Comparison against collaborative filtering and neural ranking baselines
- Online A/B testing of recommendation policies
- Contextual bandits or other sequential recommendation approaches

---

## Key Idea

The core idea of this project is to separate recommendation into two stages: retrieve relevant movies efficiently, then use reinforcement learning to construct a more diverse recommendation sequence.

This combines:

- Representation Learning
- Sequential User Modeling
- Approximate Nearest-Neighbor Retrieval
- Reinforcement Learning
- Interactive Deployment

into a single end-to-end recommendation system.


