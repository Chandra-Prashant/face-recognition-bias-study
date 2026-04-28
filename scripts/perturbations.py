import cv2
import numpy as np
import os

class FacePerturber:
    def __init__(self):
        pass

    def apply_gaussian_blur(self, image, kernel_size=7):
        """
        RQ2: Simulates motion or focus blur[cite: 21, 47].
        Proposal Levels: 3x3, 7x7, 15x15.
        """
        # Ensure kernel size is odd
        k = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        return cv2.GaussianBlur(image, (k, k), 0)

    def apply_gaussian_noise(self, image, sigma=15):
        """
        RQ2: Simulates sensor noise[cite: 21, 46].
        Proposal Levels: Sigma 5, 15, 25.
        """
        mean = 0
        # Generate noise based on the image shape
        gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
        noisy = image + gauss
        # Clip to ensure pixel values remain in [0, 255]
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_illumination_change(self, image, gamma=1.0):
        """
        RQ2: Simulates illumination variation via Gamma Correction[cite: 21, 48].
        Proposal Levels: 0.5, 0.75 (Dark) | 1.5, 2.0 (Bright).
        """
        invGamma = 1.0 / gamma
        # Build a lookup table for gamma correction
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_occlusion(self, image, landmarks, area='eyes'):
        """
        Controlled perturbation: Simulates sunglasses or masks[cite: 42, 45].
        """
        temp_img = image.copy()
        h, w, _ = temp_img.shape

        if area == 'eyes':
            le = landmarks['left_eye']
            re = landmarks['right_eye']
            start_point = (int(le[0] - w*0.1), int(le[1] - h*0.05))
            end_point = (int(re[0] + w*0.1), int(re[1] + h*0.05))
            cv2.rectangle(temp_img, start_point, end_point, (0, 0, 0), -1)

        elif area == 'mouth':
            nose = landmarks['nose']
            mouth_l = landmarks['mouth_left']
            mouth_r = landmarks['mouth_right']
            points = np.array([
                [nose[0], nose[1] - 10],
                [mouth_l[0] - 20, mouth_l[1] + 20],
                [mouth_r[0] + 20, mouth_r[1] + 20]
            ], np.int32)
            cv2.fillPoly(temp_img, [points], (0, 0, 0))

        return temp_img

# Independent verification for your report [cite: 56]
if __name__ == "__main__":
    # Create results folder if missing
    os.makedirs('results/samples', exist_ok=True)
    
    # Dummy image for testing (replace with actual path if needed)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8) + 128 
    perturber = FacePerturber()
    
    # Test all proposal-specific perturbations [cite: 45]
    cv2.imwrite('results/samples/blur_15.jpg', perturber.apply_gaussian_blur(test_img, 15))
    cv2.imwrite('results/samples/noise_25.jpg', perturber.apply_gaussian_noise(test_img, 25))
    cv2.imwrite('results/samples/gamma_0.5.jpg', perturber.apply_illumination_change(test_img, 0.5))
    
    print("✅ All proposal perturbations implemented and samples saved to results/samples/")