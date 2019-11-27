import argparse
import os
import sys

import cv2
import numpy as np

cascade_name = ""
cascade = cv2.CascadeClassifier()
groundTruth = {}
manualFaces = {
    "dart4.jpg": [(345, 100, 125, 170)],
    "dart5.jpg": [(60, 135, 60, 70), (55, 245, 60, 70), (190, 210, 60, 70), (250, 165, 55, 60), (295, 237, 50, 70), (380, 190, 60, 60), (430, 230, 55, 70), (510, 180, 60, 60), (560, 240, 55, 75), (650, 185, 55, 65), (680, 240, 50, 70)],
    "dart13.jpg": [(420, 120, 110, 140)],
    "dart14.jpg": [(470, 215, 80, 100), (735, 190, 90, 100)],
    "dart15.jpg": [(70, 125, 60, 85), (375, 110, 50, 75), (540, 125, 60, 80)]
}
manualDarts = {
    "dart0.jpg" : [(419,0,208,218)],
    "dart1.jpg" : [(166,105,256,254)],
    "dart4.jpg": [(345, 100, 125, 170)],
    "dart5.jpg": [(60, 135, 60, 70), (55, 245, 60, 70), (190, 210, 60, 70), (250, 165, 55, 60), (295, 237, 50, 70), (380, 190, 60, 60), (430, 230, 55, 70), (510, 180, 60, 60), (560, 240, 55, 75), (650, 185, 55, 65), (680, 240, 50, 70)],
    "dart13.jpg": [(420, 120, 110, 140)],
    "dart14.jpg": [(470, 215, 80, 100), (735, 190, 90, 100)],
    "dart15.jpg": [(70, 125, 60, 85), (375, 110, 50, 75), (540, 125, 60, 80)]
}

'''
Calculates intersection over union of the two rectangles and, if it is above a certain threshold, classifies them as correctly identified
facesX :: [(x,y,width,height)]
percentile
'''


def findUnionAndIntersection(detected, groundTruth, percentile):
    correctDetected = []
    correctTruths = []
    detectedSet = set(tuple(i) for i in detected)
    truthSet = set(tuple(j) for j in groundTruth)
    for truth in truthSet.symmetric_difference(correctTruths):
        for detected in detectedSet.symmetric_difference(correctDetected):
            intersectingPixelsCount = 0
            detectedSize = detected[2]*detected[3]
            trueSize = truth[2]*truth[3]
            for i in range(truth[0], truth[0]+truth[2]):
                for j in range(truth[1], truth[1]+truth[3]):
                    if (i > detected[0] and i < detected[0]+detected[2] and j > detected[1] and j < detected[1]+detected[3]):
                        intersectingPixelsCount += 1
            if (intersectingPixelsCount/(detectedSize+trueSize-intersectingPixelsCount) > percentile):
                correctDetected.append(detected)
                correctTruths.append(truth)
                break
    # Returns the TP faces and the corresponding ground truth
    return (correctDetected, correctTruths)


def calculateF1andTPR(detected, groundTruth, percentile):
    tp = findUnionAndIntersection(detected, groundTruth, percentile)[1]
    # true positive rate is true positives over all valid faces
    tpr = len(tp)/len(groundTruth)
    # precision is true positives over all detected
    precision = len(tp)/len(detected)
    if (precision + tpr) != 0:
        f1 = 2*precision*tpr/(precision+tpr)
    else:
        f1 = 0
    return f1, tpr


def detectAndDisplay(frame):
    # 1. Prepeare the image by turning it grayscale and normalising lighting.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones object detection
    detected = cascade.detectMultiScale(
        frame_gray, 1.1, 1, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))
    # 3. Print number of faces found
    print(len(detected))
    # 4. Draw green boxes around the faces found
    for (x, y, width, height) in detected:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cascade_name = os.path.basename(os.path.normpath(sys.argv[2]))
    if cascade_name == "frontalface.xml":
        groundTruth = manualFaces
    elif cascade_name == "dartboards.xml":
        print ("Detecting dartboards")
        groundTruth = manualDarts
        
    # We want to normalise the filepath, so we can understand all possible references
    normpath = os.path.basename(os.path.normpath(sys.argv[1]))
    # If we have ground truth for this file
    if (normpath in groundTruth):
        print(normpath)
        # 5. Draw red boxes around ground truth
        for (x, y, width, height) in groundTruth[normpath]:
            frame = cv2.rectangle(
                frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        # 6. Calculate TPR
        print(calculateF1andTPR(detected, groundTruth[normpath], 0.6))
    cv2.imshow('Capture - Object detection', frame)
    cv2.waitKey(0)

# Takes a single argument of the image we are trying to detect faces on


def main():
    # Read the Input Image
    frame = cv2.imread(sys.argv[1])
    # Read the cascade name
    cascade_path = sys.argv[2]


    # Load the strong classifier
    if not cascade.load(cascade_path):
        print('--(!)Error loading face cascade')
        exit(0)
    # Detect faces and display the result
    detectAndDisplay(frame)
    # Save result image
    cv2.imwrite("detected_"+sys.argv[1], frame)
    return 0


if __name__ == "__main__":
    main()
