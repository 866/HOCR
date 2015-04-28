import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/home/victor/Pictures/2015-04-28-172405.jpg", 0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
plt.imshow(img, "gray")
plt.show()