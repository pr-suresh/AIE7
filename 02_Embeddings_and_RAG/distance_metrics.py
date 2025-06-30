import numpy as np

# Euclidean Distance: sqrt(sum((a_i - b_i)²))
euclidean_distance = lambda a, b: np.sqrt(np.sum((np.array(a) - np.array(b))**2))

# Cosine Similarity: (a·b) / (||a|| ||b||)
cosine_similarity = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Manhattan Distance: sum(|a_i - b_i|)
manhattan_distance = lambda a, b: np.sum(np.abs(np.array(a) - np.array(b)))

# Example usage
if __name__ == "__main__":
    vec1, vec2 = [1, 2, 3], [4, 5, 6]
    print(f"Euclidean: {euclidean_distance(vec1, vec2):.3f}")
    print(f"Cosine: {cosine_similarity(vec1, vec2):.3f}")
    print(f"Manhattan: {manhattan_distance(vec1, vec2):.3f}")
    
    # Test with embedding-like vectors
    emb1, emb2 = [0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]
    print(f"\nEmbedding vectors:")
    print(f"Euclidean: {euclidean_distance(emb1, emb2):.3f}")
    print(f"Cosine: {cosine_similarity(emb1, emb2):.3f}")
    print(f"Manhattan: {manhattan_distance(emb1, emb2):.3f}") 