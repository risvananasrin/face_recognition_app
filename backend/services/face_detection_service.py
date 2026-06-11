from insightface.app import FaceAnalysis

class FaceDetectionService:

    def __init__(self):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image):

        if image is None:
            return []

        faces = self.app.get(image)

        # keep best faces first
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)

        return faces