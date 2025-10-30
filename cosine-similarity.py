import numpy as np
import sklearn.metrics.pairwise as pw


def cosine_similarity(vec_a, vec_b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vec_a (np.ndarray): First input vector.
    vec_b (np.ndarray): Second input vector.

    Returns:
    float: Cosine similarity between vec_a and vec_b.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def cosine_similarity_sklearn(vec_a, vec_b):
    """
    Calculate the cosine similarity between two vectors using sklearn.

    Parameters:
    vec_a (np.ndarray): First input vector.
    vec_b (np.ndarray): Second input vector.

    Returns:
    float: Cosine similarity between vec_a and vec_b.
    """
    vec_a = vec_a.reshape(1, -1)
    vec_b = vec_b.reshape(1, -1)
    
    return pw.cosine_similarity(vec_a, vec_b)[0][0] 