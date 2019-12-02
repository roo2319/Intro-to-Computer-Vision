import numpy as np
import math
import cv2

'''
Threshold is given as a number between 0 and 1. 
If a number is greater than threshold*max then it becomes 255, else 0
'''


def thresholdImage(im, threshold):
    thrVal = threshold * im.max()
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            im[y, x] = 255 if im[y, x] > thrVal else 0
    return im


def convolution(im, kernel):
    # filter 2d calculates correlation by default
    kernel = np.flipud(np.fliplr(kernel))
    return cv2.filter2D(im, -1, kernel, anchor=(-1, -1))


'''
 Assume incoming image is grayscale
'''


def sobel(im):
    # Approximate derivatives with a sobel kernel
    sobelX = np.array(([-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]))
    dX = convolution(im, sobelX)

    sobelY = np.array(([-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]))
    dY = convolution(im, sobelY)

    # Calculate the magnitude and the gradient images

    magnitude = np.zeros(im.shape)
    gradient = np.zeros(im.shape)

    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            magnitude[y, x] = math.floor(math.sqrt(dX[y, x]**2+dY[y, x]**2))
            gradient[y, x] = math.atan2(dY[y, x], dX[y, x])

    magnitude = thresholdImage(magnitude, 0.2)
    return (magnitude, gradient)


def main():
    im = cv2.imread("../test_images/ellipse.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    a = sobel(im)
    cv2.imshow("Mag", a[0])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
