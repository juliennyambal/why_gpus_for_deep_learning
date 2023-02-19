import cv2
import numpy as np

image = cv2.imread(
    "/Users/cp364698/Documents/why_gpus_for_deep_learning/1 - original_image.jpg"
)
# Get the size of image + channel size
row, col, ch = image.shape
# In order to work on the image, we need to have a placeholder of the
# same size of the image. For this instance we have an array of zeros
random_noise = np.zeros((row, col, ch), dtype=np.uint8)
# Inplace function for populating our placeholder with random value
cv2.randn(random_noise, (0, 0, 0), (255, -255, 255))
random_noise = (random_noise * 0.5).astype(np.uint8)
gn_img = cv2.add(image, random_noise)
gn_img_add = cv2.add(image, gn_img)


# cv2.imwrite("/Users/cp364698/Desktop/random_noise.jpg", random_noise)
# cv2.imwrite("/Users/cp364698/Desktop/gn_img.jpg", gn_img)
# cv2.imwrite("/Users/cp364698/Desktop/gn_img_avg.jpg", gn_img_add)
# cv2.imwrite("/Users/cp364698/Desktop/original_image.jpg", image)
