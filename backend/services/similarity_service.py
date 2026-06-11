import numpy as np

class SimilarityService:

    def compare(self, emb1, emb2):

        emb1 = np.array(emb1)
        emb2 = np.array(emb2)

        # cosine similarity (STANDARD)
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return dot / (norm1 * norm2)