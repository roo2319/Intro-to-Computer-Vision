import numpy as np
import cv2
import math
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

def distance(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

def putNumberInRange(x,a,b):
    while (x<a):
        x+=a
    while (x>=b):
        x

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

def hough(im):

    width, height = im.shape
    diagonal = int(np.ceil(np.sqrt(width*width + height*height)))
    houghSpace = np.zeros((2*diagonal, 360))

    mag, ang = sobel(im)

    # position in hough space represents location in these lists
    p = range(-diagonal, diagonal)

    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if mag[y, x] != 0:
                t = [angleD+ang[y][x] for angleD in np.arange(-0.1,0.1,0.02)]
                for t_index in range(len(t)):
                    angle = t[t_index]
                    # As range from -diagonal to diagonal,
                    # Add diagonal so range goes from 0 to 2 * diagonal
                    p_index = int(x * np.cos(angle) + y *
                                  np.sin(angle)) + diagonal
                    houghSpace[p_index, int(math.degrees(angle))%360] += 1

    for p_index in range(houghSpace.shape[0]):
        for t_index in range(houghSpace.shape[1]):
            # Hardcoded threshold, Play around (Maybe top 10?)
            if houghSpace[p_index, t_index] < 20:
                houghSpace[p_index, t_index] = 0 

    angles = []
    for p_index in range(houghSpace.shape[0]):
        for t_index in range(houghSpace.shape[1]):
            if houghSpace[p_index, t_index] > 0:
                angles.append(t_index)
                # print("start")
                angle = math.radians(t_index)
                # print(math.degrees(angle))
                distance = p_index-diagonal
                # print(distance)

                a = np.cos(angle)
                b = np.sin(angle)
                # Find base coordinates
                x0 = np.cos(angle) * distance
                y0 = np.sin(angle) * distance

                # Generate endpoints, to create a long line
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(im, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return set(map(lambda x: x//10,angles))



def findLines(image): 
    return hough(image)

def main():
    image = cv2.imread('dart2.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    findLines(frame_gray)

if __name__ == "__main__":
    main()
