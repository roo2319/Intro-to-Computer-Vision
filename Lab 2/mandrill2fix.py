import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('mandrill2.jpg',1)
cv2.imshow("Pic",img)
 
img_not = cv2.bitwise_not(img)
cv2.imshow("Invert1",img_not)
cv2.waitKey(0)
cv2.destroyAllWindows()
