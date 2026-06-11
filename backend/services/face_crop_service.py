import cv2

class FaceCropService:

    def crop_face(self, image, face):

        if face is None:
            return None

        x1, y1, x2, y2 = face.bbox.astype(int)

        # safety fix
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        return image[y1:y2, x1:x2]