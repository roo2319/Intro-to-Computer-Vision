import math

import cv2
import numpy as np

from helpers import sobel


#Function to increment array if element exists
def incIfExists(arr, ind):
    try:
        arr[ind] += 1
    except IndexError:
        pass


def hough(image,threshold):
    sobelMagnitude, sobelAngle= sobel(image)
    hough=np.zeros((len(sobelMagnitude),len(sobelMagnitude[0]),100))
    for i in range(len(sobelMagnitude)):
        for j in range(len(sobelMagnitude[0])):
            if sobelMagnitude[i,j]==255:
                for r in range(10,100):
                    x1=int(j+r*math.cos(sobelAngle[i,j]))
                    x2=int(j-r*math.cos(sobelAngle[i,j]))
                    y1=int(i+r*math.sin(sobelAngle[i,j]))
                    y2=int(i-r*math.sin(sobelAngle[i,j]))


                    incIfExists(hough,(y1,x1,r))
                    incIfExists(hough,(y1,x2,r))
                    incIfExists(hough,(y2,x1,r))
                    incIfExists(hough,(y2,x2,r))
    
    circleCount=0
    for y in range(len(hough)):
        for x in range(len(hough[0])):
            for r in range(len(hough[0,0])):
                if hough[y,x,r] > threshold:
                    circleCount+=1
                    cv2.circle(image,(x,y),r,(0),3)

    return circleCount

def findCircles(image): 
    return hough(image,15)

def main():
    image = cv2.imread('../test_images/dart1.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    findCircles(frame_gray)

if __name__ == "__main__":
    main()
