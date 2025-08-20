# Transformer

Adapted from the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

You can find the `.pdf` file of the paper from the repo

## The understanding behind the attention module

The intuition behind using the dot product in attention is fundamentally about **measuring similarity and relevance**

### The Core Intuition

Basically the dot product tells us **geometrically**, whether two vectors (matrices are vector of vectors) are pointing in the same direction or not.

The dot product between query and key vectors measures how "aligned" or similar they are. When two vectors point in similar directions, their dot product is large; when they're orthogonal, it's zero; when they point in opposite directions, it's negative.

In attention, this translates to: **"How relevant is this key-value pair to what I'm querying for?"**

### The Mathematical Flow

Here's how scaled dot-product attention works step by step:

1. **Query-Key Similarity**: `QK^T`
   - For each query vector `q` and each key vector `k`, compute their dot product `q·k`
   - This produces a matrix where entry `(i,j)` represents **how much query i "cares about" key j**
   - **Higher dot products = higher relevance**

2. **Scaling**: `QK^T / √d_k`
   - This divides our previous matmul by the square root of the key dimension
   - Tries to solve the exploding gradient problem by scaling the values down
   - Also tries to solve the vanishing gradient problem by scaling the values down after teh matmul so that the `softmax` applied doesn't make the weights diminish (sharp fall)
    - This prevents the dot products from becoming too large (which would make softmax too sharp)
    - Keeps gradients stable during training

3. **Softmax Normalization**: `softmax(QK^T / √d_k)`
   - Converts the similarity scores into a probability distribution
   - Each query now has attention weights that sum to 1 across all keys
   - This gives us "how much attention" each query pays to each key

4. **Weighted Combination**: `softmax(QK^T / √d_k)V`
   - Multiply the attention weights by the value vectors
   - This creates a weighted average of all values, where the weights come from query-key similarity

### Why This Two-Step Process?

The separation into query-key similarity followed by value weighting serves distinct purposes:

- **Keys determine relevance**: "What information is available and how relevant is it?"
- **Values contain content**: "What is the actual information to retrieve?"

This separation allows the model to:
- Use keys as "addresses" or "indices" for information retrieval
- Store the actual content in values
- Learn different representations for "what to look for" vs "what to retrieve"

### A Concrete Analogy

Think of it like a library:
- **Queries**: "I need information about photosynthesis"
- **Keys**: Book titles and subject tags (used for searching/matching)
- **Values**: The actual book contents
- **Attention weights**: **How relevant each book is to your query**
- **Output**: A summary that combines relevant books, weighted by their relevance

The dot product naturally captures this "relevance matching" because learned vector representations tend to place semantically similar concepts closer together in the vector space.

This elegant mathematical framework allows the model to dynamically decide what information to focus on based on the current context, which is the core power of the attention mechanism.
