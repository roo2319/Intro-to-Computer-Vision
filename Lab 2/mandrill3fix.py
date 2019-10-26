import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('mandrill3.jpg',1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imshow("wut",img_hsv)
cv2.waitKey(0)
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img_hsv],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# i=0
# plt.show()
# original = cv2.imread('mandrillRGB.jpg',1)
# original_hsv=cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([original_hsv],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()