import numpy as np
import cv2
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.draw import circle_perimeter

def convolution(xs,ys):
    result=0
    m=len(xs)
    n=len(xs[0])
    for i in range(m):
        for j in range(n):
            result+=xs[m-i-1,n-j-1]*ys[i,j]
    return result

def mirrorNumberInRange(x,start,end):
    if x<start:
        return start+x
    elif x>=end:
        return end-1-(end-x)
    else:
         return x

# uses mirorring for edge cases
def findImageMatrix(image, xCenter , yCenter, kernelLength, kernelWidth):
    imageMatrix = np.zeros((kernelLength,kernelWidth))
    for i in range(0, kernelLength):
        for j in range(0,kernelWidth):
            x=int(xCenter-((kernelLength-1)/2)+i)
            y=int(yCenter-((kernelWidth-1)/2)+j)
            imageMatrix[i,j]=image[mirrorNumberInRange(x,0,len(image)), mirrorNumberInRange(y,0,len(image[0]))]
    return imageMatrix


def applyKernel(kernel,image):
    alteredImage=np.zeros((len(image),len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[0])):
            imageMatrix=findImageMatrix(image, i, j, len(kernel), len(kernel[1]))
            alteredImage[i,j]=int(convolution(imageMatrix,kernel))
    return(alteredImage)

def findMagnitude(matrixA, matrixB):
    if (len(matrixA)!=len(matrixB) or len(matrixA[0])!=len(matrixB[0])):
        raise Exception("matrices are not the same size")
    magnitude=np.zeros((len(matrixA),len(matrixA[0])))
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            magnitude[i,j]= int(math.sqrt(matrixA[i,j]**2+matrixB[i,j]**2))
    return magnitude

def findGradient(matrixA, matrixB):
    if (len(matrixA)!=len(matrixB) or len(matrixA[0])!=len(matrixB[0])):
        raise Exception("matrices are not the same size")
    gradient=np.zeros((len(matrixA),len(matrixA[0])))
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            gradient[i,j]=math.atan2(matrixB[i,j], matrixA[i,j])
    return gradient

def thresholdImage(image,thershold):
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i,j]>thershold:
                image[i,j]=255
            else:
                image[i,j]=0
    return image

def sobel(image):
    # image = cv2.medianBlur(image, 5)
    kernelX=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
    alteredImageX=applyKernel(kernelX,image)
    kernelY=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    alteredImageY=applyKernel(kernelY,image)
    magnitude=findMagnitude(alteredImageX,alteredImageY)
    magnitude=np.interp(magnitude, (magnitude.min(), magnitude.max()), (0, 1))
    gradient=findGradient(alteredImageX,alteredImageY)
    thresholdedImage=thresholdImage(magnitude,0.4)
    cv2.imshow("edgedetectionGradientThresholded",thresholdedImage)
    cv2.waitKey(0)
    return (magnitude,gradient)

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
    for i in range (len(hough)):
        for j in range (len(hough[0])):
            for k in range (len(hough[0][0])):
                if (hough[i,j,k]>0):
                    cv2.circle(image,(j,i),k,(255),1)
    cv2.imshow("aaaa", image)
    cv2.waitKey(0)
    return hough

def findCircles(image): 
    return hough(image,15)

def main():
    image = cv2.imread('dart2.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    findCircles(frame_gray)

if __name__ == "__main__":
    main()