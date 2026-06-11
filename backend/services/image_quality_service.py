import cv2
import numpy as np

class ImageQualityService:

    def is_valid_image(self, image):

        if image is None:
            print("DEBUG: Image is None")
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        print("DEBUG Brightness:", brightness)
        print("DEBUG Blur:", blur)

        # relaxed thresholds (IMPORTANT FIX)
        if brightness < 20 or brightness > 250:
            print("FAILED: brightness")
            return False

        # FIXED blur threshold (this was your problem)
        if blur < 15:
            print("FAILED: blur")
            return False

        return True