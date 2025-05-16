import cv2

class ProcessingImage:
    """
    This is a Image Processing class that runs the processes:
        - Convert image to grayscale.
        - Apply Gaussian Filter.
        - Apply Binary Threshold with optimum threshold value.

    Args:
        image: 
    """
    def __init__(self, input_image):
        self.input_image = input_image

    def convert_image_to_gray(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def apply_gaussian_blurring_to_image(self,image):
        blur_image = cv2.GaussianBlur(image, (5,5), 0)
        return blur_image
    
    def apply_binary_threshold(self,image):
        threshold_value_otsu, threshold_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold_image
    
    def run(self):
        """
        Executes all image processing methods.

        Return: Opencv image processed....
        """
        gray_image = self.convert_image_to_gray(self.input_image)
        blur_image = self.apply_gaussian_blurring_to_image(gray_image)
        binary_image = self.apply_gaussian_blurring_to_image(blur_image)
        
        return binary_image