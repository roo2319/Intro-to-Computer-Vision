import numpy as np
import cv2
import math
import random
from scipy import ndimage
import struct
import random
#from skimage import transform


def convolution(xs, ys):
    result = 0
    m = len(xs)
    n = len(xs[0])
    for i in range(m):
        for j in range(n):
            result += xs[m-i-1, n-j-1]*ys[i, j]
    return result


def mirrorNumberInRange(x, start, end):
    if x < start:
        return start+x
    elif x >= end:
        return end-1-(end-x)
    else:
         return x

# uses mirorring for edge cases


def findImageMatrix(image, xCenter, yCenter, kernelLength, kernelWidth):
    imageMatrix = np.zeros((kernelLength, kernelWidth))
    for i in range(0, kernelLength):
        for j in range(0, kernelWidth):
            x = int(xCenter-((kernelLength-1)/2)+i)
            y = int(yCenter-((kernelWidth-1)/2)+j)
            imageMatrix[i, j] = image[mirrorNumberInRange(
                x, 0, len(image)), mirrorNumberInRange(y, 0, len(image[0]))]
    return imageMatrix


def applyKernel(kernel, image):
    alteredImage = np.zeros((len(image), len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[0])):
            imageMatrix = findImageMatrix(
                image, i, j, len(kernel), len(kernel[1]))
            alteredImage[i, j] = int(convolution(imageMatrix, kernel))
    return(alteredImage)


def findMagnitude(matrixA, matrixB):
    if (len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0])):
        raise Exception("matrices are not the same size")
    magnitude = np.zeros((len(matrixA), len(matrixA[0])))
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            magnitude[i, j] = int(math.sqrt(matrixA[i, j]**2+matrixB[i, j]**2))
    return magnitude


def findGradient(matrixA, matrixB):
    if (len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0])):
        raise Exception("matrices are not the same size")
    gradient = np.zeros((len(matrixA), len(matrixA[0])))
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            gradient[i, j] = math.atan2(matrixB[i, j], matrixA[i, j])
    return gradient


def thresholdImage(image, thershold):
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i, j] > thershold:
                image[i, j] = 255
            else:
                image[i, j] = 0
    return image


def calculateMean(image):
    return(ndimage.measurements.center_of_mass(image))


def calculateGeneralisedHoughSpace(image, gradient, threshold):
    mean = calculateMean(image)
    xRef = int(mean[0])
    yRef = int(mean[1])
    accumulator = np.array(360)

# Find the distance between two tuples of points


def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def putNumberInRange(x, a, b):
    while (x < a):
        x += a
    while (x >= b):
        x


def sobel(image):
    image = cv2.medianBlur(image, 5)
    kernelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    alteredImageX = applyKernel(kernelX, image)
    kernelY = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    alteredImageY = applyKernel(kernelY, image)
    magnitude = findMagnitude(alteredImageX, alteredImageY)
    magnitude = np.interp(
        magnitude, (magnitude.min(), magnitude.max()), (0, 1))
    gradient = findGradient(alteredImageX, alteredImageY)
    thresholdedImage = thresholdImage(magnitude, 0.2)
    return (magnitude, gradient)


def houghEllipses(im, minDistance, threshold):
    # 1. Store all the edge pixels in a 1D array
    validEllipses = []
    pixels = []
    for i in range(len(im)):
        for j in range(len(im[0])):
            if im[i, j] == 255 and random.randint(1,10) == 2:
                pixels.append((j, i))
    print (len(pixels))

    #Keeps 20% of pixels randomly to reduce runtime
    pixels=random.sample(pixels, int(len(pixels)/5))

    #2. For each pixel, carry out the following
    for p1 in pixels:

        # 3. For each other pixel (if distance is more than min distance)
        candidatePixels = [p for p in pixels if distance(
            p, p1) > minDistance and p != p1]
        # print("P1 loop")
        for p2 in candidatePixels:

            # 4. Clear the accumulator array
            accumulator = np.zeros(300)

            # 5. Calculate the *center*, length of major *axis* and *orientation* of the assumed ellipse

            center = ((p1[0]+p2[0])/2, (p1[1]+p2[1]))
            axis = distance(p1, p2)/2
            if p2[0] - p1[0] == 0:
                break
            orientation = math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))

            # 6. For each candidate pixel that is not p2
            for p3 in [p for p in candidatePixels if p != p2]:
                # (The entire thing is in a try catch loop, to handle edge cases of approximation)
                try:
                    # 7. If the *dist*ance is greater than the minimum distance, Calculate the minor axis (*maxis*)
                    dist = distance(center, p3)
                    if (dist > minDistance):
                        cosTao = (axis**2 + dist**2 -
                                  distance(p3, p2)**2)/(2*axis*dist)
                        maxis = math.ceil(
                            math.sqrt((axis**2 * dist**2 * (1-cosTao**2))/(axis**2 - dist**2 * cosTao**2)))

                        if maxis < len(accumulator):
                            accumulator[maxis] += 1
                except:
                    continue

            # 8. If the max is greater than the theshold, then we have an ellipse!
            if (accumulator.max() >= threshold):
                validEllipses.append(((int(center[0]),int(center[1])), (int(axis), int(maxis)), int(orientation)))
                pixels.remove(p1)
                pixels.remove(p2)
                pixels.remove(p3)
                del accumulator
                print("ellipse found!")
                break
        
    return validEllipses

def detectEllipses(image):
    sobelMagnitude, sobelAngle = sobel(image)
    ellipses = houghEllipses(sobelMagnitude, 40, 250)
    print(ellipses)
    for (center,axis,orientation) in ellipses:
        cv2.ellipse(image, center, axis, orientation, 0, 360, (200,50,255))
    cv2.imshow("Ellipses", image)
    return len(ellipses)

def main():
    image = cv2.imread("dart2.jpg")
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    ellipses = detectEllipses("dart2.jpg")
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
