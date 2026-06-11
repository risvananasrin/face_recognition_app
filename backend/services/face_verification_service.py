import numpy as np
from insightface.app import FaceAnalysis

class FaceVerificationService:
    def __init__(self, threshold=0.48):
        self.threshold = threshold

        # Load InsightFace model
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, img):
        faces = self.app.get(img)

        if len(faces) == 0:
            return None

        # Take the first detected face
        return faces[0].embedding

    def cosine_similarity(self, emb1, emb2):
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        return np.dot(emb1, emb2)

    def verify(self, img1, img2):

        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)

        if emb1 is None or emb2 is None:
            return {
                "match_found": False,
                "similarity_score": 0.0,
                "message": "No face detected in one or both images"
            }

        # similarity score (0 to 1 approx)
        score = float(self.cosine_similarity(emb1, emb2))

        # IMPORTANT FIX: similarity logic
        match = score >= self.threshold

        return {
            "match_found": match,
            "similarity_score": score,
            "threshold": self.threshold
        }