import numpy as np
import cv2
import math

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

image = cv2.imread('coins1.png',0)
kernelX=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
alteredImageX=cv2.convertScaleAbs(applyKernel(kernelX,image))
cv2.imshow("edgedetectionX",alteredImageX)
cv2.waitKey(0)
kernelY=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
alteredImageY=cv2.convertScaleAbs(applyKernel(kernelY,image))
cv2.imshow("edgedetectionY",alteredImageY)
cv2.waitKey(0)
magnitude=findMagnitude(alteredImageX,alteredImageY)
cv2.imshow("edgedetectionMagnitude",magnitude)
cv2.waitKey(0)
gradient=findGradient(alteredImageX,alteredImageY)
cv2.imshow("edgedetectionGradient",gradient)
cv2.waitKey(0)