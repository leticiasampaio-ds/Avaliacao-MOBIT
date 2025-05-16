import cv2

from image_processing import ProcessingImage
from contours import Contours

# Read image file
image = cv2.imread(r'assets/raw-image.png')
# Processed image
processed_image = ProcessingImage(image).run()
# Find and Draw image external contours
contours, image_with_contours = Contours(processed_image).run()
# Print the number of detected/contourned objects
print(f'Number of detected objects {len(contours)}.')

# Display the output image
cv2.imshow('Output Image', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()