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


def hough(image):
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
    for y in range(len(hough)):
        for x in range(len(hough[0])):
            for r in range(len(hough[0,0])):
                if hough[y,x,r] < 10:
                    hough[y,x,r] = 0
                
    return hough


def findCircles(image): 
    houghs = hough(image)
    circleCount=0
    for y in range(len(houghs)):
        for x in range(len(houghs[0])):
            for r in range(len(houghs[0,0])):
                if houghs[y,x,r] != 0:
                    circleCount+=1
    # cv2.imshow("circles",image)
    # cv2.waitKey(0)
    return circleCount

def findBestCircle(image):
    houghs = hough(image)
    y,x,r = np.unravel_index(np.argmax(houghs, axis=None), houghs.shape)
    while y-r<0 or x-r<0 or x+r>len(houghs[0]) or y+r>len(houghs):
        houghs[y,x,r] = 0
        y,x,r = np.unravel_index(np.argmax(houghs, axis=None), houghs.shape)
    # y,x = np.unravel_index(np.argmax(np.sum(houghs,axis=2)),houghs.shape[:2])
    # print(houghs[y,x])
    # r = np.argmax(houghs[y,x])
    cv2.circle(image,(x,y),r,0,2)
    cv2.imshow("a",image)
    cv2.waitKey(0)
    return (x-r,y-r, 2*r, 2*r)



def main():
    image = cv2.imread('../test_images/dart0.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)
    findCircles(frame_gray)

if __name__ == "__main__":
    main()
