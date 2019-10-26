import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('mandrill0.jpg',1)
# show histograms
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# i=0
# plt.show()
# original = cv2.imread('mandrillRGB.jpg',1)
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([original],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
red = img[:,:,2].copy()
green = img[:,:,1].copy()
blue = img[:,:,0].copy()

img[:,:,0]=red
img[:,:,1]=blue
img[:,:,2]=green

cv2.imshow("window",img)
cv2.waitKey(0)