import cv2

from image_processing import ProcessingImage

image = cv2.imread(r'assets/raw-image.png')
processed_image = ProcessingImage(image).run()

cv2.imshow('image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()