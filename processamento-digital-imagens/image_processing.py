import cv2
import numpy as np

class ProcessingImage:
    """
    Image preprocessing pipeline class.
    
    Applies the following sequential operations:
        - Conversion to grayscale.
        - Gaussian blurring for noise reduction.
        - Binary thresholding using Otsu's method.

    Attributes:
        input_image (np.ndarray): Original BGR image loaded with OpenCV.
    """
    def __init__(self, input_image: np.ndarray):
        self.input_image = input_image

    def convert_image_to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Converts a BGR image to grayscale.

        Args:
            image (np.ndarray): The input color image.

        Returns:
            np.ndarray: Grayscale image (1 channel).
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def apply_gaussian_blurring_to_image(self, image: np.ndarray, kernel_size=(5, 5)) -> np.ndarray:
        """
        Applies a Gaussian blur to reduce image noise and detail.

        Args:
            image (np.ndarray): Grayscale image.
            kernel_size (tuple): Size of the Gaussian kernel. Default is (5, 5).

        Returns:
            np.ndarray: Blurred image.
        """
        blur_image = cv2.GaussianBlur(image, kernel_size, 0)
        return blur_image

    def apply_binary_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Otsu's binary thresholding to separate foreground from background.

        Args:
            image (np.ndarray): Blurred grayscale image.

        Returns:
            np.ndarray: Binary image (pixel values are 0 or 255).
        """
        threshold_value_otsu, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold_image

    def run(self) -> np.ndarray:
        """
        Runs the complete preprocessing pipeline:
            - Convert to grayscale
            - Apply Gaussian blur
            - Apply binary thresholding

        Returns:
            np.ndarray: Preprocessed binary image ready for contour detection.
        """
        gray_image = self.convert_image_to_gray(self.input_image)
        blur_image = self.apply_gaussian_blurring_to_image(gray_image)
        binary_image = self.apply_binary_threshold(blur_image)
        return binary_image