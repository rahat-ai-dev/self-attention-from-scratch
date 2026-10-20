"""
Sentence -> word vectors -> Q, K, V -> relation between every pair of words.

Run:
    python word_vectors_and_relations.py "The cat drank milk"
or just:
    python word_vectors_and_relations.py
and type the sentence when asked.

NOTE: Embedding and W_q / W_k / W_v are RANDOM here (no training), so the
relations you see are NOT real language meaning. They only show HOW the
numbers work. Change SEED to get a different random result.
"""
import sys
import numpy as np

SEED = 1
EMBED_SIZE = 8   # numbers per word embedding (X)
QK_SIZE = 4      # numbers per Q / K / V vector
ZERO_BAND = 0.5  # |score| smaller than this is called "about zero"

np.set_printoptions(precision=2, suppress=True, linewidth=150)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def label(score):
    if score > ZERO_BAND:
        return "POSITIVE"
    if score < -ZERO_BAND:
        return "NEGATIVE"
    return "ZERO"


# ---------- 1. get the sentence ----------
if len(sys.argv) > 1:
    sentence = " ".join(sys.argv[1:])
else:
    sentence = input("Write a sentence: ")

tokens = sentence.split()
n = len(tokens)
if n == 0:
    sys.exit("Empty sentence.")

# ---------- 2. word -> ID -> embedding vector (X) ----------
rng = np.random.default_rng(SEED)
vocab = {}                      # word -> ID
for w in tokens:
    if w.lower() not in vocab:
        vocab[w.lower()] = len(vocab)

embedding_table = rng.standard_normal((len(vocab), EMBED_SIZE))  # one row per unique word
ids = [vocab[w.lower()] for w in tokens]
X = embedding_table[ids]        # shape (n, EMBED_SIZE)

print("=" * 70)
print("STEP 1: word -> ID")
for w, i in zip(tokens, ids):
    print(f"  {w:<12} -> ID {i}")

print("\nSTEP 2: embedding vector X (each row = one word)")
for w, row in zip(tokens, X):
    print(f"  {w:<12} {row}")

# ---------- 3. Q, K, V ----------
W_q = rng.standard_normal((EMBED_SIZE, QK_SIZE)) * 0.5
W_k = rng.standard_normal((EMBED_SIZE, QK_SIZE)) * 0.5
W_v = rng.standard_normal((EMBED_SIZE, QK_SIZE)) * 0.5
Q, K, V = X @ W_q, X @ W_k, X @ W_v

print("\n" + "=" * 70)
print("STEP 3: Query vectors  Q = X @ W_q")
for w, row in zip(tokens, Q):
    print(f"  {w:<12} {row}")
print("\n        Key vectors    K = X @ W_k")
for w, row in zip(tokens, K):
    print(f"  {w:<12} {row}")
print("\n        Value vectors  V = X @ W_v")
for w, row in zip(tokens, V):
    print(f"  {w:<12} {row}")

# ---------- 4. relation: Query of word i vs Key of word j ----------
scores = Q @ K.T / np.sqrt(QK_SIZE)   # scaled dot product

print("\n" + "=" * 70)
print("STEP 4: score table  (row = Query word, column = Key word)")
print("        score = (Q . K) / sqrt(d_k)")
w_col = max(len(t) for t in tokens) + 2
print(" " * (w_col + 2) + "".join(f"{t:>{w_col}}" for t in tokens))
for i, t in enumerate(tokens):
    print(f"  {t:<{w_col}}" + "".join(f"{scores[i, j]:>{w_col}.2f}" for j in range(n)))

print("\n" + "=" * 70)
print(f"STEP 5: relation of every pair  (ZERO means |score| < {ZERO_BAND})")
for i in range(n):
    for j in range(n):
        s = scores[i, j]
        print(f"  Query '{tokens[i]}'  vs  Key '{tokens[j]}'  ->  {s:>6.2f}   {label(s)}")
    print()

# ---------- 5. attention weights (softmax) ----------
weights = softmax(scores)
print("=" * 70)
print("STEP 6: attention weights after softmax (each row sums to 1)")
print(" " * (w_col + 2) + "".join(f"{t:>{w_col}}" for t in tokens))
for i, t in enumerate(tokens):
    print(f"  {t:<{w_col}}" + "".join(f"{weights[i, j]:>{w_col}.2f}" for j in range(n)))

print("\nWho does each word look at most?")
for i, t in enumerate(tokens):
    j = int(weights[i].argmax())
    print(f"  '{t}' pays most attention to '{tokens[j]}' ({weights[i, j]:.2f})")

output = weights @ V
print("\n" + "=" * 70)
print("STEP 7: final output = weights @ V (new vector for each word)")
for w, row in zip(tokens, output):
    print(f"  {w:<12} {row}")
