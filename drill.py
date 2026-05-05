"""
Module 6 Week B — Core Skills Drill: Embedding Basics

Complete the three functions below to load, query, and compare
pre-trained GloVe word embeddings.
"""

import numpy as np


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array of shape (50,).
    """
    embeddings = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 51:  # word + 50 dimensions
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:51]], dtype=np.float32)
                    embeddings[word] = vector
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        return {}
    except Exception as e:
        print(f"Error loading GloVe file: {e}")
        return {}
    
    return embeddings


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1]. If either vector has zero norm, return 0.0.
    """
    # Calculate norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero norm case
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    cosine_sim = dot_product / (norm1 * norm2)
    
    return float(cosine_sim)


def nearest_neighbors(word, embeddings, n=5):
    """Find the n most similar words to the given word.

    Returns a list of (word, score) tuples sorted by similarity descending,
    excluding the query word itself.
    """
    if word not in embeddings:
        return []
    
    query_vector = embeddings[word]
    similarities = []
    
    # Calculate similarity with all other words
    for other_word, other_vector in embeddings.items():
        if other_word != word:  # Exclude the query word itself
            sim = cosine_similarity(query_vector, other_vector)
            similarities.append((other_word, sim))
    
    # Sort by similarity in descending order and return top n
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


if __name__ == "__main__":
    glove = load_glove("data/glove_50k_50d.txt")
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
