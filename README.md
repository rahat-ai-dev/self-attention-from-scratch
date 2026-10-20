# Self-Attention From Scratch (NumPy)

Type any sentence and see how every word relates to every other word inside self-attention: **positive**, **negative**, or **zero**.

Built with plain NumPy. No deep learning library does the work.

## The question

When a model reads *"My name is Rahat"*, how does each word decide which other words to look at?

This project answers that by showing every number involved, step by step.

## What the code does

1. Takes a sentence you type.
2. Gives each word an ID and turns it into a **vector** (a list of numbers). This is called an embedding.
3. Creates a **Query**, **Key** and **Value** vector for each word.
   - Query: what this word is looking for
   - Key: the label other words match against
   - Value: the information this word shares
4. Scores every pair of words with a scaled dot product: `score = (Q · K) / √d_k`
5. Labels each pair:
   - **POSITIVE**: the vectors point the same way (strong link)
   - **NEGATIVE**: the vectors point opposite ways (weak link)
   - **ZERO**: no clear link (score between -0.5 and 0.5)
6. Applies **softmax** to turn scores into attention weights (each row adds up to 1).
7. Mixes the Value vectors using those weights to produce the final output.

The core formula:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

## How to run

You only need Python 3 and NumPy.

```bash
pip install numpy
python word_vectors_and_relations.py "My name is Rahat"
```

Or run it without a sentence and type one when asked:

```bash
python word_vectors_and_relations.py
```

## Example output

Sentence: `My name is Rahat`

Score table (row = Query word, column = Key word):

```
              My   name     is  Rahat
  My        0.11   0.47   0.48  -0.85
  name     -0.03   0.26  -0.33  -0.69
  is        0.52  -1.36  -0.17   2.57
  Rahat    -0.49  -1.08   0.52   2.37
```

Relations for the word `is`:

```
  Query 'is'  vs  Key 'My'     ->   0.52   POSITIVE
  Query 'is'  vs  Key 'name'   ->  -1.36   NEGATIVE
  Query 'is'  vs  Key 'is'     ->  -0.17   ZERO
  Query 'is'  vs  Key 'Rahat'  ->   2.57   POSITIVE
```

Attention weights after softmax:

```
              My   name     is  Rahat
  My        0.24   0.34   0.34   0.09
  name      0.28   0.37   0.21   0.14
  is        0.11   0.02   0.05   0.82
  Rahat     0.05   0.03   0.13   0.80
```

So `is` gives 82% of its attention to `Rahat`.

## Important note: the weights are random

The embedding table and the matrices `W_q`, `W_k`, `W_v` are **random and not trained**. That means:

- The relations you see are **not real language meaning**.
- The demo shows **how** the numbers work, not **what** words mean.
- Change `SEED` at the top of the script and the whole result changes.

In a real model, these weights are learned from huge amounts of text. Training is what makes related words (like "cat" and "drank") end up with vectors that score high together.

## Things I learned

- A model does not "understand" words. It compares numbers using dot products.
- Dividing by `√d_k` keeps softmax from collapsing onto a single word when vectors are large.
- Softmax never gives a true zero, so even an unrelated word gets a small share of attention.
- A negative score means "weak link", and softmax turns it into a very small weight.

## Settings you can change

At the top of `word_vectors_and_relations.py`:

| Setting | Meaning | Default |
|---|---|---|
| `SEED` | Random seed. Change it to get different vectors | `1` |
| `EMBED_SIZE` | Numbers per word embedding | `8` |
| `QK_SIZE` | Numbers per Query / Key / Value vector | `4` |
| `ZERO_BAND` | Scores smaller than this (in absolute value) are labelled ZERO | `0.5` |

## Next steps

- [ ] Use pretrained embeddings (for example GloVe) so relations carry real meaning
- [ ] Check the output against PyTorch's `F.scaled_dot_product_attention`
- [ ] Add a heatmap of the attention weights
- [ ] Extend to multi-head attention

## Author

Rahat ([@rahat-ai-dev](https://github.com/rahat-ai-dev))

Part of my self-study journey in AI/ML. Feedback and your own test sentences are welcome.
