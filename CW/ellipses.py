import numpy as np
import cv2
import math
from scipy import ndimage
import struct
from skimage import transform
import random

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

def distance(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

def putNumberInRange(x,a,b):
    while (x<a):
        x+=a
    while (x>=b):
        x

def sobel(image):
    image = cv2.medianBlur(image, 5)
    kernelX=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
    alteredImageX=applyKernel(kernelX,image)
    kernelY=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    alteredImageY=applyKernel(kernelY,image)
    magnitude=findMagnitude(alteredImageX,alteredImageY)
    magnitude=np.interp(magnitude, (magnitude.min(), magnitude.max()), (0, 1))
    gradient=findGradient(alteredImageX,alteredImageY)
    thresholdedImage=thresholdImage(magnitude,0.3)
    cv2.imshow("edgedetectionGradientThresholded",thresholdedImage)
    cv2.waitKey(0)
    return (magnitude,gradient)

def houghEllipses(im,minDistance, threshold):
    pixels=[]
    for i in range(len(im)):
        for j in range(len(im[0])):
            if im[i,j]==255:
                pixels.append((j,i))
    pixels=random.sample(pixels,int(len(pixels)/20))
    validEllipses=[]
    for (x1,y1) in pixels:
        # print("outer loop")
        # print(x1,y1)
        for (x2,y2) in [p2 for p2 in pixels if p2!= (x1,y1)]:
            # print("inner loop")
            # print(x2,y2)
            if (distance(x1,y1,x2,y2)>minDistance and x1!=x2):
                accumulator=np.zeros(300)
                x0=(x1+x2)/2
                y0=(y1+y2)/2
                a=distance(x1,y1,x2,y2)/2
                alpha=math.atan((y2-y1)/(x2-x1))
                for (x3,y3) in [p3 for p3 in pixels if (p3!= (x1,y1) and p3!= (x2,y2))]:
                    d=distance(x0,y0,x3,y3)
                    if (d>minDistance):
                        cosTao=(a**2 + d**2 - distance(x3,y3,x2,y2)**2)/(2*a*d)
                        if (a**2 - d**2 * cosTao**2)!=0:
                            bSquared=(a**2 * d**2 * (1-cosTao**2))/(a**2 - d**2 * cosTao**2)
                            if bSquared>0:
                                b=int(math.sqrt(bSquared))
                                if b<len(accumulator):
                                    accumulator[b]+=1
                if (accumulator.max()>=threshold):
                    print("found")
                    print(accumulator.max())
                    validEllipses.append((x0,y0,a,b,alpha))
                    pixels[:] = [p for p in pixels if (p!=(x1,y1) and p!=(x2,y2) and p!=(x3,y3))]
                    del accumulator
                    break
        # else: 
        #     continue
        # break
    return validEllipses

def main(): 
    image = cv2.imread('dart1.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    sobelMagnitude, sobelAngle= sobel(frame_gray)
    ellipses=houghEllipses(sobelMagnitude, 40, 16)
    # ellipses = transform.hough_ellipse(sobelMagnitude, accuracy=20, threshold=250,
    #                    min_size=100, max_size=120)
    for (x,y,a,b,alpha) in ellipses:
        cv2.ellipse(image, (int(x),int(y)), (int(a),int(b)), alpha, 365, 0, (200,50,255))
    cv2.imshow("aaaa", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
