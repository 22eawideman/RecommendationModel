import pandas as pd
import numpy as np

df = pd.read_csv('/Users/ethanwideman/Downloads/ml-1m/ratings.dat', sep='::', engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'])

df['user_id'] = pd.Categorical(df['user_id']).codes
item_cat = pd.Categorical(df['item_id'])
item_id_map = dict(enumerate(item_cat.categories))
df['item_id'] = item_cat.codes

df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
test_idx = df.groupby('user_id')['timestamp'].idxmax()
test_df = df.loc[test_idx]
train_df = df.drop(test_idx)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class MatrixFactorization:
  def __init__(self, n_users, n_items, n_factors=64, lr=0.001):
    self.user_embeddings = np.random.normal(0, 0.01, (n_users, n_factors))
    self.item_embeddings = np.random.normal(0, 0.01, (n_items, n_factors))
    self.lr = lr

  def forward(self, user_ids, item_ids):
    u = self.user_embeddings[user_ids]
    v = self.item_embeddings[item_ids]
    return np.sum(u * v, axis=1)

  def loss(self, predictions, labels):
    preds = sigmoid(predictions)
    return -np.mean(
      labels * np.log(preds + 1e-9) +
      (1 - labels) * np.log(1 - preds + 1e-9)
    )

  def backward(self, user_ids, item_ids, predictions, labels):
    error = sigmoid(predictions) - labels
    u = self.user_embeddings[user_ids]
    v = self.item_embeddings[item_ids]
    np.add.at(self.user_embeddings, user_ids, -self.lr * error[:, None] * v)
    np.add.at(self.item_embeddings, item_ids, -self.lr * error[:, None] * u)


class NCF:
  def __init__(self, n_users, n_items, n_factors=64, hidden=[128, 64], lr=0.001,
               beta1=0.9, beta2=0.999, eps=1e-8):
    self.user_embeddings = np.random.normal(0, 0.01, (n_users, n_factors))
    self.item_embeddings = np.random.normal(0, 0.01, (n_items, n_factors))
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.t = 0

    layer_sizes = [2 * n_factors] + hidden + [1]
    self.weights = []
    self.biases = []
    self.mW = []
    self.vW = []
    self.mb = []
    self.vb = []

    self.m_user = np.zeros_like(self.user_embeddings)
    self.v_user = np.zeros_like(self.user_embeddings)
    self.m_item = np.zeros_like(self.item_embeddings)
    self.v_item = np.zeros_like(self.item_embeddings)

    for i in range(len(layer_sizes) - 1):
      fan_in = layer_sizes[i]
      W = np.random.normal(0, np.sqrt(2 / fan_in), (layer_sizes[i], layer_sizes[i + 1]))
      b = np.zeros(layer_sizes[i + 1])
      self.weights.append(W)
      self.biases.append(b)
      self.mW.append(np.zeros_like(W))
      self.vW.append(np.zeros_like(W))
      self.mb.append(np.zeros_like(b))
      self.vb.append(np.zeros_like(b))

  def forward(self, user_ids, item_ids):
    u = self.user_embeddings[user_ids]
    v = self.item_embeddings[item_ids]
    self.cache = {'z': [], 'a': [np.concatenate([u, v], axis=1)]}
    for i, (W, b) in enumerate(zip(self.weights, self.biases)):
      z = self.cache['a'][-1] @ W + b
      self.cache['z'].append(z)
      a = np.maximum(0, z) if i < len(self.weights) - 1 else sigmoid(z)
      self.cache['a'].append(a)
    return self.cache['a'][-1].squeeze()

  def score_with_embedding(self, user_emb, item_ids):
    u = np.tile(user_emb, (len(item_ids), 1))
    v = self.item_embeddings[item_ids]
    a = np.concatenate([u, v], axis=1)
    for i, (W, b) in enumerate(zip(self.weights, self.biases)):
      z = a @ W + b
      a = np.maximum(0, z) if i < len(self.weights) - 1 else sigmoid(z)
    return a.squeeze()

  def loss(self, predictions, labels):
    preds = np.clip(predictions, 1e-9, 1 - 1e-9)
    return -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))

  def backward(self, user_ids, item_ids, predictions, labels):
    batch_size = len(labels)
    dout = ((predictions - labels) / batch_size)[:, None]

    grad_weights = []
    grad_biases = []

    for i in reversed(range(len(self.weights))):
      a_prev = self.cache['a'][i]
      z = self.cache['z'][i]
      if i < len(self.weights) - 1:
        dout = dout * (z > 0)
      grad_weights.insert(0, a_prev.T @ dout)
      grad_biases.insert(0, dout.sum(axis=0))
      dout = dout @ self.weights[i].T

    self.t += 1
    bc1 = 1 - self.beta1 ** self.t
    bc2 = 1 - self.beta2 ** self.t

    for i in range(len(self.weights)):
      self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * grad_weights[i]
      self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * grad_weights[i] ** 2
      self.weights[i] -= self.lr * (self.mW[i] / bc1) / (np.sqrt(self.vW[i] / bc2) + self.eps)

      self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * grad_biases[i]
      self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * grad_biases[i] ** 2
      self.biases[i] -= self.lr * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + self.eps)

    n_factors = self.user_embeddings.shape[1]
    grad_u = dout[:, :n_factors]
    grad_v = dout[:, n_factors:]

    np.add.at(self.m_user, user_ids, (1 - self.beta1) * (grad_u - self.m_user[user_ids]))
    np.add.at(self.v_user, user_ids, (1 - self.beta2) * (grad_u ** 2 - self.v_user[user_ids]))
    np.add.at(self.m_item, item_ids, (1 - self.beta1) * (grad_v - self.m_item[item_ids]))
    np.add.at(self.v_item, item_ids, (1 - self.beta2) * (grad_v ** 2 - self.v_item[item_ids]))

    m_u_hat = self.m_user[user_ids] / bc1
    v_u_hat = self.v_user[user_ids] / bc2
    m_i_hat = self.m_item[item_ids] / bc1
    v_i_hat = self.v_item[item_ids] / bc2

    np.add.at(self.user_embeddings, user_ids, -self.lr * m_u_hat / (np.sqrt(v_u_hat) + self.eps))
    np.add.at(self.item_embeddings, item_ids, -self.lr * m_i_hat / (np.sqrt(v_i_hat) + self.eps))


def sample_negatives(train_df, n_items, n_neg=4):
  user_item_set = set(zip(train_df['user_id'], train_df['item_id']))
  pos_users = train_df['user_id'].values
  pos_items = train_df['item_id'].values
  n_pos = len(pos_users)

  neg_users = np.repeat(pos_users, n_neg)
  neg_items = np.random.randint(0, n_items, size=n_pos * n_neg)

  mask = np.array([(u, i) in user_item_set for u, i in zip(neg_users, neg_items)])
  neg_items[mask] = np.random.randint(0, n_items, size=mask.sum())

  users = np.concatenate([pos_users, neg_users])
  items = np.concatenate([pos_items, neg_items])
  labels = np.concatenate([np.ones(n_pos), np.zeros(n_pos * n_neg)])
  return users, items, labels.astype(float)


def train(model, train_df, n_items, epochs=20, batch_size=256, n_neg=4):
  for epoch in range(epochs):
    users, items, labels = sample_negatives(train_df, n_items, n_neg)
    idx = np.random.permutation(len(users))
    users, items, labels = users[idx], items[idx], labels[idx]

    total_loss = 0
    n_batches = len(users) // batch_size

    for b in range(n_batches):
      sl = slice(b * batch_size, (b + 1) * batch_size)
      preds = model.forward(users[sl], items[sl])
      total_loss += model.loss(preds, labels[sl])
      model.backward(users[sl], items[sl], preds, labels[sl])

    print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / n_batches:.4f}")


def evaluate(model, train_df, test_df, n_items, K=10, n_neg=99):
  user_item_set = set(zip(train_df['user_id'], train_df['item_id']))
  item_counts = train_df['item_id'].value_counts()

  model_hits, model_ndcgs = [], []
  pop_hits, pop_ndcgs = [], []

  for _, row in test_df.iterrows():
    user = int(row['user_id'])
    pos_item = int(row['item_id'])

    negatives = []
    while len(negatives) < n_neg:
      neg = np.random.randint(0, n_items)
      if (user, neg) not in user_item_set and neg != pos_item:
        negatives.append(neg)

    candidates = np.array([pos_item] + negatives)
    users = np.full(len(candidates), user)

    scores = model.forward(users, candidates)
    ranked = candidates[np.argsort(-scores)]
    if pos_item in ranked[:K]:
      rank = np.where(ranked == pos_item)[0][0] + 1
      model_hits.append(1)
      model_ndcgs.append(1 / np.log2(rank + 1))
    else:
      model_hits.append(0)
      model_ndcgs.append(0)

    pop_scores = np.array([item_counts.get(i, 0) for i in candidates])
    pop_ranked = candidates[np.argsort(-pop_scores)]
    if pos_item in pop_ranked[:K]:
      rank = np.where(pop_ranked == pos_item)[0][0] + 1
      pop_hits.append(1)
      pop_ndcgs.append(1 / np.log2(rank + 1))
    else:
      pop_hits.append(0)
      pop_ndcgs.append(0)

  print(f"Model      — HR@{K}: {np.mean(model_hits):.4f}  NDCG@{K}: {np.mean(model_ndcgs):.4f}")
  print(f"Popularity — HR@{K}: {np.mean(pop_hits):.4f}  NDCG@{K}: {np.mean(pop_ndcgs):.4f}")


def build_title_map(movies_path, item_id_map):
  movies = pd.read_csv(movies_path, sep='::', engine='python',
                       names=['movie_id', 'title', 'genres'], encoding='latin-1')
  orig_to_title = dict(zip(movies['movie_id'], movies['title']))
  return {code: orig_to_title[orig] for code, orig in item_id_map.items()
          if orig in orig_to_title}


def get_recommendations(user_id, model, code_to_title, watched_codes, n=10):
  all_items = np.arange(model.item_embeddings.shape[0])
  users = np.full(len(all_items), user_id)
  scores = model.forward(users, all_items)
  scores[list(watched_codes)] = -np.inf
  top_idx = np.argsort(-scores)[:n]
  return [(code_to_title.get(i, f'movie_{i}'), float(scores[i])) for i in top_idx]


def recommend_from_history(seed_codes, model, code_to_title, n=10):
  user_emb = model.item_embeddings[seed_codes].mean(axis=0)
  all_items = np.arange(model.item_embeddings.shape[0])
  scores = model.score_with_embedding(user_emb, all_items)
  scores[seed_codes] = -np.inf
  top_idx = np.argsort(-scores)[:n]
  return [(code_to_title.get(i, f'movie_{i}'), float(scores[i])) for i in top_idx]


def plot_growing_history(seed_codes, model, code_to_title, history_sizes=[1, 3, 5, 10]):
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(2, 2, figsize=(16, 10))
  axes = axes.flatten()

  for ax, size in zip(axes, history_sizes):
    seeds = seed_codes[:size]
    recs = recommend_from_history(seeds, model, code_to_title, n=10)
    titles = [r[0][:45] for r in recs]
    scores = [r[1] for r in recs]
    ax.barh(range(10), scores[::-1], color='steelblue')
    ax.set_yticks(range(10))
    ax.set_yticklabels(titles[::-1], fontsize=8)
    ax.set_xlabel('Score')
    ax.set_title(f'After watching {size} movie{"s" if size > 1 else ""}')

  seed_titles = [code_to_title.get(c, f'movie_{c}') for c in seed_codes[:3]]
  fig.suptitle(f'Recommendations growing from:\n{", ".join(seed_titles)}', fontsize=11)
  plt.tight_layout()
  plt.savefig('recommendations_demo.png', dpi=150, bbox_inches='tight')
  plt.show()
  print("Saved to recommendations_demo.png")


n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()

model = NCF(n_users, n_items, n_factors=64, hidden=[128, 64], lr=0.001)
train(model, train_df, n_items, epochs=40)
evaluate(model, train_df, test_df, n_items)

movies_df = pd.read_csv('/Users/ethanwideman/Downloads/ml-1m/movies.dat', sep='::', engine='python',
                        names=['movie_id', 'title', 'genres'], encoding='latin-1')
code_to_title = build_title_map('/Users/ethanwideman/Downloads/ml-1m/movies.dat', item_id_map)
orig_to_genres = dict(zip(movies_df['movie_id'], movies_df['genres']))

action_codes = [code for code, orig in item_id_map.items()
                if 'Action' in orig_to_genres.get(orig, '')]
action_counts = train_df[train_df['item_id'].isin(action_codes)]['item_id'].value_counts()
seed_codes = action_counts.index[:10].tolist()

plot_growing_history(seed_codes, model, code_to_title)
