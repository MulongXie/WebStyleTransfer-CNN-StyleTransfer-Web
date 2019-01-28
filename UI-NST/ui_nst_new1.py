import cv2
import numpy as np

img = np.ones((20, 40, 3))
print(np.shape(img))

img[:, :, 0] = 255
img[:, :, 1] = 0
img[:, :, 2] = 0

cv2.imshow('img', img)
cv2.imwrite("con_button2.jpg", img)
cv2.waitKey(0)
