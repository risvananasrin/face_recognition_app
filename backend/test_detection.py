import cv2

from services.face_verification_service import FaceVerificationService

verifier = FaceVerificationService()

img1 = cv2.imread("test_images/IMG_3150.JPG")
img2 = cv2.imread("test_images/r.jpg")

result = verifier.verify(img1, img2)

print(result)