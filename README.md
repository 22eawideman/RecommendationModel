# Streamlair Recommendation AI

A video recommendation system built from scratch using NumPy. Implements Neural Collaborative Filtering (NCF) — the same approach described in YouTube's original recommendation paper — without PyTorch or TensorFlow. Every forward pass, loss calculation, and gradient is implemented by hand.

## What It Does

Takes a user's watch history and predicts what videos they're likely to watch next. This is a collaborative filtering problem — it learns patterns from user behavior, not video content.

Each user and each video gets a learned embedding vector. The model predicts watch likelihood by computing the dot product of a user's vector and a video's vector, then refining that prediction through a small neural network.

## Model Architecture

**Phase 1 — Matrix Factorization**

The foundation. Each user and item gets an `n_factors`-dimensional embedding (default: 64). The dot product of user × item gives a raw score. Trained with binary cross-entropy and negative sampling.

**Phase 2 — Neural Collaborative Filtering (NCF)**

A small MLP sits on top of the concatenated user and item embeddings:

```
[user_embedding | item_embedding] → Dense(128) → ReLU → Dense(64) → ReLU → Dense(1) → Sigmoid
```

Every layer's forward and backward pass is implemented manually using the chain rule — no autograd.

## Math Implemented From Scratch

- Forward pass (dot product of embeddings)
- Binary cross-entropy loss
- Backpropagation through each MLP layer
- Adam optimizer
- Negative sampling (randomly sampled unwatched items as negatives)

## Dataset

**MovieLens 1M** — 1 million ratings, ~6,000 users, ~4,000 movies.

- Download: [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens)
- Format: `userId::movieId::rating::timestamp`
- Movies serve as a stand-in for videos — same collaborative filtering problem

May scale to MovieLens 25M later.

## Data Pipeline

IDs are remapped to 0-indexed integers before splitting (raw MovieLens IDs have gaps).

**Train/test split:** Leave-one-out per user — each user's most recent interaction is held out as the test item, everything before it is training data. This matches the original NCF paper's evaluation protocol, enabling direct comparison to published benchmarks.

```python
df['user_id'] = pd.Categorical(df['user_id']).codes
df['item_id'] = pd.Categorical(df['item_id']).codes
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
test_idx = df.groupby('user_id')['timestamp'].idxmax()
test_df  = df.loc[test_idx]
train_df = df.drop(test_idx)
```

## Evaluation

- **Hit Rate@K** — did the actual held-out item appear in the top K recommendations
- **NDCG@K** — normalized discounted cumulative gain, rewards ranking the correct item higher
- Compared against a popularity baseline (recommend most-watched items)

## Tech Stack

| Library | Role |
|---|---|
| numpy | All model math — embeddings, forward pass, gradients |
| pandas | Data loading and preprocessing |
| matplotlib | Visualization |
| scikit-learn | Evaluation metrics and data utilities only |

No PyTorch or TensorFlow. NumPy forces every gradient to be written by hand, which is the point.

## Implementation Phases

1. **Data pipeline** — load MovieLens, remap IDs, leave-one-out split, negative sampling
2. **Matrix Factorization** — embeddings, dot product forward pass, BCE loss, manual backprop
3. **NCF** — MLP on top of concatenated embeddings, all gradients by hand
4. **Evaluation** — Hit Rate@K and NDCG@K vs popularity baseline
5. **Demo** — top N recommendations from a simulated watch history, matplotlib visualization

## Future Plans

- MLX version for Apple Silicon GPU acceleration once the NumPy version is complete
- Web portfolio page to showcase alongside Streamlair

## Context

This project complements [Streamlair](https://streamlair.net), a full-stack YouTube-like video platform (Next.js, Express, PostgreSQL, AWS). The recommendation system demonstrates ML fundamentals to go alongside the platform's full-stack and DevOps work.
