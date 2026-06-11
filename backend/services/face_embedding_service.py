class FaceEmbeddingService:

    def get_embedding(self, face):

        if face is None:
            return None

        return face.embedding