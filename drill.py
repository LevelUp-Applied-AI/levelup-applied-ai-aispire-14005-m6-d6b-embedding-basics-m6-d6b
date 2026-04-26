"""
Module 6 Week B — Core Skills Drill: Embedding Basics

Complete the three functions below to load, query, and compare
pre-trained GloVe word embeddings.
"""

import numpy as np


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Each line in the file has the format: word dim1 dim2 ... dimN

    Args:
        filepath: Path to the GloVe text file (e.g., 'data/glove_5k_50d.txt').

    Returns:
        Dictionary mapping each word (str) to its embedding (numpy array of shape (50,)).
    """
    # TODO: Read the file line by line, split each line into word and vector,
    #       and store in a dictionary
    embeddings = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector (numpy array).
        vec2: Second vector (numpy array).

    Returns:
        Float: cosine similarity score in [-1, 1].
    """
    # TODO: Compute the dot product divided by the product of norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def nearest_neighbors(word, embeddings, n=5):
    """Find the n most similar words to the given word in embedding space.

    Args:
        word: Query word (str).
        embeddings: Dictionary mapping words to numpy arrays (from load_glove).
        n: Number of neighbors to return.

    Returns:
        List of (word, similarity_score) tuples, sorted by similarity descending.
        The query word itself should NOT appear in the results.
    """
    # TODO: Compute cosine similarity between the query word and all other words,
    #       then return the top-n most similar
    query_vec = embeddings[word]
    scores = []
    for other_word, other_vec in embeddings.items():
        if other_word == word:
            continue
        sim = cosine_similarity(query_vec, other_vec)
        scores.append((other_word, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


if __name__ == "__main__":
    glove = load_glove("data/glove_5k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} word vectors")

        # Task 2: Word similarity
        sim = cosine_similarity(glove.get("king", np.zeros(50)),
                                glove.get("queen", np.zeros(50)))
        if sim is not None:
            print(f"cosine('king', 'queen') = {sim:.4f}")

        sim2 = cosine_similarity(glove.get("king", np.zeros(50)),
                                 glove.get("banana", np.zeros(50)))
        if sim2 is not None:
            print(f"cosine('king', 'banana') = {sim2:.4f}")

        # Task 3: Nearest neighbors
        neighbors = nearest_neighbors("king", glove, n=5)
        if neighbors:
            print(f"Nearest to 'king': {neighbors}")

        print("testing done!")
