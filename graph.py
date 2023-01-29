import cv2
import numpy as np
img=cv2.imread('ziwen_lu_0002_aligned.png')

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)

