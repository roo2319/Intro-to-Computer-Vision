import numpy as np
import cv2

img = cv2.imread('mandrill.jpg')
for i in range(len(img)):
    for j in range(len(img[0])):
        # for nose
        if (img[i,j][0]>108 and img[i,j][0]<115):   
          img[i,j]=[255,255,255]
        else: img[i,j]=[0,0,0]
cv2.imshow("white",img)
cv2.waitKey(0)