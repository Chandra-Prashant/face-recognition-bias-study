import cv2
import numpy as np
import os

class FacePerturber:
    def __init__(self):
        pass

    def apply_gaussian_blur(self, image, kernel_size=(15, 15)):
        """Simulates low-quality sensor or motion blur."""
        return cv2.GaussianBlur(image, kernel_size, 0)

    def apply_brightness_change(self, image, value=-50):
        """Simulates poor lighting conditions (underexposure)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v.astype(int) + value, 0, 255).astype('uint8')
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def apply_occlusion(self, image, landmarks, area='eyes'):
        """
        Simulates sunglasses or masks using facial landmarks.
        landmarks: dict from RetinaFace (e.g., {'left_eye': (x, y), ...})
        """
        temp_img = image.copy()
        h, w, _ = temp_img.shape

        if area == 'eyes':
            # Draw a black rectangle across the eyes (Sunglasses simulation)
            le = landmarks['left_eye']
            re = landmarks['right_eye']
            # Calculate a region covering both eyes
            start_point = (int(le[0] - w*0.1), int(le[1] - h*0.05))
            end_point = (int(re[0] + w*0.1), int(re[1] + h*0.05))
            cv2.rectangle(temp_img, start_point, end_point, (0, 0, 0), -1)

        elif area == 'mouth':
            # Draw a polygon covering the nose and mouth (Mask simulation)
            nose = landmarks['nose']
            mouth_l = landmarks['mouth_left']
            mouth_r = landmarks['mouth_right']
            
            # Simple triangle/polygon mask
            points = np.array([
                [nose[0], nose[1] - 10],
                [mouth_l[0] - 20, mouth_l[1] + 20],
                [mouth_r[0] + 20, mouth_r[1] + 20]
            ], np.int32)
            cv2.fillPoly(temp_img, [points], (0, 0, 0))

        return temp_img

# Example Usage Implementation
if __name__ == "__main__":
    # This is for testing the script independently
    img = cv2.imread('data/LFW/lfw-deepfunneled/lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg')
    perturber = FacePerturber()
    
    # Apply blur
    blurred = perturber.apply_gaussian_blur(img)
    
    # Save or Display to verify
    cv2.imwrite('results/sample_blur.jpg', blurred)
    print("Perturbation samples saved to results/")