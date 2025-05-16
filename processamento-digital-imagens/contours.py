import cv2
import numpy as np

class Contours:
    """
    This class is responsible for finding and drawing external object contours
    from a binary image.

    Args:
        image (np.ndarray): The input image, expected to be binary (1-channel).
    """
    def __init__(self, image: np.ndarray):
        self.image = image

    def find_external_contours(self) -> list:
        """
        Finds the external contours in the binary image using OpenCV's findContours.

        Returns:
            list: A list of contours found in the image. Each contour is a NumPy array of points.
        """
        contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_contours_by_area(self, contours: list, min_area: int = 100) -> list:
        """
        Filters contours by minimum area to remove small noise.

        Args:
            contours (list): List of contours to filter.
            min_area (int): Minimum area threshold. Contours with smaller area will be ignored.

        Returns:
            list: Filtered list of contours.
        """
        return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    def draw_image_contours(self, contours: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Draws the given contours on a copy of the image (converted to BGR).

        Args:
            contours (list): List of contours to draw.
            color (tuple): BGR color for the contour lines.
            thickness (int): Thickness of the contour lines.

        Returns:
            np.ndarray: A 3-channel image (BGR) with the contours drawn.
        """
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_rgb, contours, -1, color, thickness)
        return image_rgb

    def run(self) -> tuple:
        """
        Runs the full contour detection, filtering, and drawing pipeline.

        Args:
            min_area (int): Minimum area threshold for valid contours.

        Returns:
            tuple:
                - contours (list): List of filtered contours.
                - image_with_contours (np.ndarray): Image with contours drawn in color.
        """
        contours = self.find_external_contours()
        filtered_contours = self.filter_contours_by_area(contours)
        image_with_contours = self.draw_image_contours(filtered_contours)
        return filtered_contours, image_with_contours