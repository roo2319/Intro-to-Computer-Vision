import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from helpers import sobel


def hough(image,threshold):
    sobelMagnitude, sobelAngle= sobel(image)
    hough=np.zeros((len(sobelMagnitude)+100,len(sobelMagnitude[0])+100,200))
    for i in range(len(sobelMagnitude)):
        for j in range(len(sobelMagnitude[0])):
            if sobelMagnitude[i,j]==255:
                for r in range(20,100):
                    x1=int(j+r*math.cos(sobelAngle[i,j]))
                    x2=int(j-r*math.cos(sobelAngle[i,j]))
                    y1=int(i+r*math.sin(sobelAngle[i,j]))
                    y2=int(i-r*math.sin(sobelAngle[i,j]))
                    hough[y1,x1,r]+=1
                    hough[y1,x2,r]+=1
                    hough[y2,x1,r]+=1
                    hough[y2,x2,r]+=1
    for i in range(len(hough)):
        for j in range(len(hough[0])):
            for r in range(len(hough[0,0])):
                if hough[i,j,r]<threshold:
                    hough[i,j,r]=0
    flag=0
    for i in range (len(hough)):
        for j in range (len(hough[0])):
            for k in range (len(hough[0][0])):
                if (hough[i,j,k]>0):
                    flag+=1
                    cv2.circle(image,(j,i),k,(255),1)
    cv2.imshow("Circles", image)
    cv2.waitKey(0)
    return flag

def findCircles(image): 
    return hough(image,15)

def main():
    image = cv2.imread('dart1.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    findCircles(frame_gray)

if __name__ == "__main__":
    main()