from __future__ import print_function
import cv2
import argparse
import matplotlib.pyplot as plt
import sys
import os

cascade_name = "cascade.xml"
cascade = cv2.CascadeClassifier()


def detectAndDisplay(frame):

    # 1. Prepeare the image by turning it grayscale and
    #    normalising lighting.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # 2. Perform Viola-Jones object detection
    faces = cascade.detectMultiScale(
        frame_gray, 1.1, 1, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))

    # 3. Print the number of faces found
    print(len(faces))

    # 4. Draw boxes around the faces found
    for (x, y, width, height) in faces:
        frame = cv2.rectangle(
            frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

    cv2.imshow('Object detection', frame)
    cv2.waitKey(0)


# Takes a single argument of the image we are trying to detect faces on
def main():

    # 1. Read the Input Image
    frame = cv2.imread(sys.argv[1])

    # 2. Load the strong classifier
    if not cascade.load(cascade_name):
        print('--(!)Error loading face cascade')
        exit(0)

    # 3. Detect faces and display the result
    detectAndDisplay(frame)

    # 4. Save result image
    cv2.imwrite("detected.jpg", frame)

    return 0


if __name__ == "__main__":
    main()
