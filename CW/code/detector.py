import os
import sys

import cv2
import numpy as np

import circlesusinggradient
import ellipses
import lineswithgradient
import stage4


cascade_path = "../cascades/cascade.xml"
cascade = cv2.CascadeClassifier()
manualDarts = {
    "dart0.jpg": [(444, 15, 150, 177)],
    "dart1.jpg": [(196, 133, 195, 190)],
    "dart2.jpg": [(103, 98, 86, 87)],
    "dart3.jpg": [(325, 149, 64, 69)],
    "dart4.jpg": [(184, 94, 203, 203)],
    "dart5.jpg": [(433, 141, 106, 106)],
    "dart6.jpg": [(212, 118, 61, 61)],
    "dart7.jpg": [(255, 170, 145, 145)],
    "dart8.jpg": [(842, 217, 115, 121), (62, 253, 62, 87)],
    "dart9.jpg": [(199, 49, 232, 232)],
    "dart10.jpg": [(92, 105, 95, 108), (584, 128, 55, 82), (917, 151, 34, 62)],
    "dart11.jpg": [(176, 104, 55, 74)],
    "dart12.jpg": [(156, 78, 60, 135)],
    "dart13.jpg": [(272, 122, 131, 130)],
    "dart14.jpg": [(121, 101, 124, 126), (989, 96, 122, 124)],
    "dart15.jpg": [(155, 56, 131, 138)]
}


'''
Calculates intersection over union of the two rectangles and, if it is above a certain threshold, classifies them as correctly identified
facesX :: [(x,y,width,height)]
percentile
'''

f = open("tprf1.txt", "w+")
f1total = 0
falsepositives = 0
truepositives = 0


def findUnionAndIntersection(detected, manualDarts, percentile):
    correctDetected = []
    correctTruths = []
    IOU = []
    detectedSet = set(tuple(i) for i in detected)
    truthSet = set(tuple(j) for j in manualDarts)

    # Unpack rectangle
    for (xt, yt, wt, ht) in truthSet.symmetric_difference(correctTruths):
        for (xd, yd, wd, hd) in detectedSet.symmetric_difference(correctDetected):
            intersectingPixelsCount = 0
            detectedSize = wd*hd
            trueSize = wt*ht
            for i in range(xt, xt+wt):
                for j in range(yt, yt+ht):
                    if (i > xd and i < xd+wd and j > yd and j < yd+hd):
                        intersectingPixelsCount += 1

            IOU.append(intersectingPixelsCount /
                       (detectedSize+trueSize-intersectingPixelsCount))

            if IOU[-1] > percentile:
                correctDetected.append((xd, yd, wd, hd))
                correctTruths.append((xt, yt, wt, ht))
                break
    # Returns the TP objects and the corresponding ground truth
    global falsepositives
    falsepositives += len(detected)-len(correctDetected)
    global truepositives
    truepositives += len(correctDetected)
    return correctDetected, correctTruths, IOU


def calculateF1andTPR(detected, manualDarts, percentile):
    _, tp, iou = findUnionAndIntersection(detected, manualDarts, percentile)

    print("IOU Values: {}".format(iou))
    # true positive rate is true positives over all valid objects
    tpr = len(tp)/len(manualDarts)
    # precision is true positives over all detected
    if len(detected) != 0:
        precision = len(tp)/len(detected)
    else:
        precision = 0
    if (precision + tpr) != 0:
        f1 = 2*precision*tpr/(precision+tpr)
    else:
        f1 = 0
    print(tpr)
    global f1total
    f1total += f1
    print("F1: {}, TPR: {}".format(f1, tpr))
    return f1, tpr


def fixRange(n, min, max):
    if n < min:
        return min
    if n > max:
        return max
    return n


def detectAndDisplay(frame, name):
    # 1. Prepeare the image by turning it grayscale and normalising lighting.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray_equalised = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones object detection
    detected = cascade.detectMultiScale(
        frame_gray, 1.01, 1, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))
    # # 3. Combine rectangles then print number of objects found
    # detected = combineRectangles(detected)
    print("Initial detected: {}".format(len(detected)))

    # 4. Draw green boxes around the objects found
    refined = []
    for (x, y, width, height) in detected:

        numberOfLines = len(lineswithgradient.findLines(
            frame_gray[y:y+height, x:x+width]))
        if (numberOfLines > 5):
            numberOfEllipses = len(ellipses.detectEllipses(frame_gray[fixRange(y-20, 0, len(frame_gray)):fixRange(
                y+height+20, 0, len(frame_gray)), fixRange(x-20, 0, len(frame_gray[0])):fixRange(x+height+20, 0, len(frame_gray[0]))]))
            
            if numberOfEllipses >= 1:
            # if 1:
                cv2.rectangle(frame, (x, y), (x + width,
                                              y + height), (0, 255, 0), 2)
                refined.append((x, y, width, height))

    # The best cirlce
    # bonus = circlesusinggradient.findBestCircle(frame_gray)
    # x, y, width, height = bonus
    # print(x,y,width,height)
    # if len(lineswithgradient.findLines(
    #         frame_gray[y:y+height, x:x+width])) > 5:
    #     cv2.rectangle(frame, (x, y), (x + width,
    #                                 y + height), (0, 255, 0), 2)
    #     refined.append((x, y, width, height))
    #     print("Kept bonus")

    # We want to normalise the filepath, so we can understand all possible references
    normpath = os.path.basename(os.path.normpath(name))
    # If we have ground truth for this file
    if (normpath in manualDarts):
        print(normpath)
        # 5. Draw red boxes around ground truth
        for (x, y, width, height) in manualDarts[normpath]:
            frame = cv2.rectangle(
                frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        # 6. Calculate TPR
        f.write(str(calculateF1andTPR(refined, manualDarts[normpath], 0.5)))
    print("Number of detected dartboards: {}".format(len(refined)))
    cv2.imshow('Capture - Object detection', frame)
    cv2.waitKey(0)

# Takes a single argument of the image we are trying to detect objects on


def main():

    # Load the strong classifier
    if not cascade.load(cascade_path):
        print('--(!)Error loading object cascade')
        exit(0)

    try:
        # Read the Input Image
        frame = cv2.imread(sys.argv[1])
        detectAndDisplay(frame, sys.argv[1])
        cv2.imwrite("detected_"+sys.argv[1], frame)
    except:
        # Run for all images if no second arg
        print("Running on all images")
        for name in manualDarts.keys():
            frame = cv2.imread("../test_images/"+name)
            detectAndDisplay(frame, name)
    global f1total
    print(f1total/16)
    print("false positives:")
    print(falsepositives)
    print("true positives:")
    print(truepositives)
    # Detect objects and display the result
    return 0


if __name__ == "__main__":
    main()
