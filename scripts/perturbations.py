import cv2
import numpy as np
import os

class FacePerturber:
    def __init__(self):
        pass

    def apply_gaussian_blur(self, image, kernel_size=7):
        """Simulates motion or focus blur."""
        k = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        return cv2.GaussianBlur(image, (k, k), 0)

    def apply_gaussian_noise(self, image, sigma=15):
        """Simulates sensor noise."""
        mean = 0
        gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_illumination_change(self, image, gamma=1.0):
        """Simulates illumination variation via Gamma Correction."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_occlusion(self, image, landmarks, area='eyes'):
        """Standard occlusion for Sunglasses/Masks."""
        temp_img = image.copy()
        h, w, _ = temp_img.shape

        if area == 'eyes':
            le = landmarks['left_eye']
            re = landmarks['right_eye']
            # Increased padding for better landmark coverage
            start_point = (int(le[0] - w*0.12), int(le[1] - h*0.08))
            end_point = (int(re[0] + w*0.12), int(re[1] + h*0.08))
            cv2.rectangle(temp_img, start_point, end_point, (0, 0, 0), -1)

        elif area == 'mouth':
            mouth_l = landmarks['mouth_left']
            mouth_r = landmarks['mouth_right']
            # Rectangle mask for mouth to simulate medical masks
            start_point = (int(mouth_l[0] - 10), int(mouth_l[1] - 20))
            end_point = (int(mouth_r[0] + 10), int(mouth_r[1] + 30))
            cv2.rectangle(temp_img, start_point, end_point, (0, 0, 0), -1)

        return temp_img

    # NEW: Causal Evidence / XAI Function
    def generate_occlusion_patch(self, image, x, y, patch_size=40):
        """
        Creates a sliding window occlusion to test local sensitivity.
        This provides the 'causal evidence' requested by reviewers.
        """
        temp_img = image.copy()
        cv2.rectangle(temp_img, (x, y), (x + patch_size, y + patch_size), (0, 0, 0), -1)
        return temp_img

if __name__ == "__main__":
    os.makedirs('results/samples', exist_ok=True)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8) + 128 
    perturber = FacePerturber()
    
    cv2.imwrite('results/samples/blur_15.jpg', perturber.apply_gaussian_blur(test_img, 15))
    cv2.imwrite('results/samples/noise_25.jpg', perturber.apply_gaussian_noise(test_img, 25))
    cv2.imwrite('results/samples/gamma_0.5.jpg', perturber.apply_illumination_change(test_img, 0.5))
    
    print("✅ All perturbations including XAI hooks implemented.")