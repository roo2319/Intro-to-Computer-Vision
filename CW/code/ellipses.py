import math
import random

import cv2
import numpy as np

from helpers import sobel


def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def houghEllipses(im, minDistance, threshold):
    # 1. Store all the edge pixels in a 1D array
    validEllipses = []
    pixels = []
    for i in range(len(im)):
        for j in range(len(im[0])):
            if im[i, j] == 255 and random.randint(1,10) == 2:
                pixels.append((j, i))


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

            center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
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
                validEllipses.append(((int(center[0]),int(center[1])), (int(axis), int(maxis)), int(math.degrees(orientation))))
                pixels.remove(p1)
                pixels.remove(p2)
                pixels.remove(p3)
                del accumulator
                break
        
    return validEllipses

def detectEllipses(image):
    sobelMagnitude, sobelAngle = sobel(image)
    cv2.imshow("why ruairi", sobelMagnitude)
    cv2.waitKey(0)
    ellipses = houghEllipses(sobelMagnitude, 40, 30)
    return ellipses

def main():
    image = cv2.imread('../test_images/ellipse.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)
    ellipses = detectEllipses(frame_gray)
    for (center,axis,orientation) in ellipses:
        print(center)
        print(axis)
        print(orientation)
        cv2.ellipse(image, center, axis, orientation, 0, 360, (200,50,255))
    cv2.imshow("Ellipses", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
