import numpy as np
import cv2
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.draw import circle_perimeter
from scipy import ndimage
import struct

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

def calculateMean(image):
    return(ndimage.measurements.center_of_mass(image))

def calculateGeneralisedHoughSpace(image, gradient,threshold):
    mean=calculateMean(image)
    xRef=int(mean[0])
    yRef=int(mean[1])
    accumulator=np.array(360)

def distance(a,b):
    return int(math.sqrt(a*a+b*b))

#how is there no good library function for this?
def mod(a,b):
    while a<0:
        a+=b
    while a>=b:
        a-=b
    return a

def calculateLineHoughSpace(image,gradient,threshold):
    print(2*distance(len(image),len(image[0])))
    hough=np.zeros(2*(distance(len(image),len(image[0])),180))
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i,j]==255:
                for theta in range(mod(int(math.degrees(gradient[i,j])-10),180), mod(int(math.degrees(gradient[i,j])+10),180)):
                    r=int(j*math.cos(math.radians(theta))+i*math.sin(math.radians(theta)))
                    print((len(hough)/2 + r))
                    hough[(len(hough)/2 + r),int(theta)]+=1
    for i in range(len(hough)):
        for j in range(len(hough[0])):
            if hough[i,j]<threshold:
                hough[i,j]=0
            else:
                hough[i,j]=255
    return hough

def main():
    print(struct.calcsize("P") * 8)
    image = cv2.imread('line.png',0)
    image = cv2.medianBlur(image, 5)
    kernelX=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
    alteredImageX=applyKernel(kernelX,image)
    kernelY=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    alteredImageY=applyKernel(kernelY,image)
    magnitude=findMagnitude(alteredImageX,alteredImageY)
    magnitude=np.interp(magnitude, (magnitude.min(), magnitude.max()), (0, 1))
    gradient=findGradient(alteredImageX,alteredImageY)
    print(gradient.min())
    print(gradient.max())
    thresholdedImage=thresholdImage(magnitude,0.2)
    cv2.imshow("edgedetectionGradientThresholded",thresholdedImage)
    cv2.waitKey(0)
    hough=calculateLineHoughSpace(thresholdedImage,gradient,10)
    cv2.imshow("linehough",hough)
    cv2.waitKey(0)
    for i in range(len(hough)):
        for j in range(len(hough[0])):
            if hough[i,j]==255:
                rho=i
                theta=math.radians(j)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(image,(x1,y1),(x2,y2),(255),2)
    print("got here?")
    cv2.imshow('houghlines3',image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
