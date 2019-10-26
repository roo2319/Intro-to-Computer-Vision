import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('mandrill1.jpg',1)
for i in range(len(img)-1,0,-1):
    for j in range(len(img[0])-1,0,-1):
        x=i-32
        if (x<0):
            x+=512
        y=j-32
        if(y<0):
            y+=512
        img[i,j,2]=img[x,y,2]
cv2.imshow("ifthisworksi'llbesohappy", img)
cv2.waitKey(0)