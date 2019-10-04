import numpy as np
import cv2

img = cv2.imread('mandrillRGB.jpg')
for i in range(len(img)):
  for j in range(len(img[0])):
    # detect cheeks
    # if (not (img[i,j][0]>210 and img[i,j][1]<210 and img[i,j][2]<185)):
    #       img[i,j]=[0,0,0]
    # detect nose
    # if (not (img[i,j][0]<150 and img[i,j][1]<90 and img[i,j][2]>185)):
    #       img[i,j]=[0,0,0]
    # detect eyes
    if (not (img[i,j][0]<50 and img[i,j][1]>100 and img[i,j][2]>150 and img[i,j][2]<220)):
          img[i,j]=[0,0,0]
cv2.imshow("white",img)
cv2.waitKey(0)