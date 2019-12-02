import os
import sys

import cv2
import numpy as np


cascade_path = "frontalface.xml"
cascade = cv2.CascadeClassifier()
manualFaces = {
    "dart4.jpg": [(345, 100, 125, 170)],
    "dart5.jpg": [(60, 135, 60, 70), (55, 245, 60, 70), (190, 210, 60, 70), (250, 165, 55, 60), (295, 237, 50, 70), (380, 190, 60, 60), (430, 230, 55, 70), (510, 180, 60, 60), (560, 240, 55, 75), (650, 185, 55, 65), (680, 240, 50, 70)],
    "dart13.jpg": [(420, 120, 110, 140)],
    "dart14.jpg": [(470, 215, 80, 100), (735, 190, 90, 100)],
}


'''
Calculates intersection over union of the two rectangles and, if it is above a certain threshold, classifies them as correctly identified
facesX :: [(x,y,width,height)]
percentile
'''


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
        frame_gray, 1.1, 1, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))
    # # 3. Combine rectangles then print number of objects found
    # detected = combineRectangles(detected)

    # 4. Draw green boxes around the objects found
    for (x, y, width, height) in detected:
        cv2.rectangle(frame, (x, y), (x + width,
                                      y + height), (0, 255, 0), 2)
    # We want to normalise the filepath, so we can understand all possible references
    normpath = os.path.basename(os.path.normpath(name))
    # If we have ground truth for this file
    if (normpath in manualFaces):
        # 5. Draw red boxes around ground truth
        for (x, y, width, height) in manualFaces[normpath]:
            frame = cv2.rectangle(
                frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
    print("Number of detected faces: {}".format(len(detected)))
    cv2.imshow('Capture - Object detection', frame)
    cv2.waitKey(0)

# Takes a single argument of the image we are trying to detect objects on


def main():

    # Read the cascade name
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
        for name in os.listdir("test_images"):
            frame = cv2.imread("test_images/"+name)
            detectAndDisplay(frame, name)
    # Detect objects and display the result
    return 0


if __name__ == "__main__":
    main()
