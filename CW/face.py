from __future__ import print_function
import cv2
import argparse
import matplotlib.pyplot as plt
import sys
import numpy as np

face_cascade_name = "frontalface.xml"
face_cascade = cv2.CascadeClassifier()




def detectAndDisplay(frame):
    # 1. Prepeare the image by turning it grayscale and normalising lighting.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    haar=cv2.CASCADE_SCALE_IMAGE
    # 2. Perform Viola-Jones object detection
    faces = face_cascade.detectMultiScale(frame_gray,1.1,1,0|haar, (50, 50), (500,500))
    # 3. Print number of faces found
    print(len(faces))
    # 4. Draw green boxes around the faces found
    for (x,y,width,height) in faces:
        frame = cv2.rectangle(frame, (x,y), (x + width, y + height), ( 0, 255, 0 ), 2)
    # Ground truth
    groundTruth={
        "dart4.jpg": [(345,100,125,170)],
        "dart5.jpg": [(60,135,60,70),(55,245,60,70),(190,210,60,70),(250,165,55,60),(295,237,50,70), (380,190,60,60), (430,230,55,70), (510,180,60,60), (560,240,55,75), (650,185,55,65), (680,240,50,70)],
        "dart13.jpg": [(420,120,110,140)],
        "dart14.jpg": [(470,215,80,100), (735,190,90,100)],
        "dart15.jpg": [(70,125,60,85),(375,110,50,75),(540,125,60,80)]
        }
    # If we have ground truth for this file
    if (sys.argv[1] in groundTruth):
        #5. Draw red boxes around ground truth faces
        for (x,y,width,height) in groundTruth[sys.argv[1]]:
            frame = cv2.rectangle(frame, (x,y), (x + width, y + height), ( 0, 0, 255 ), 2)
        # 6. Calculate F1 and TPR
        print(calculateF1andTPR(faces, groundTruth[sys.argv[1]], 0.6))
    cv2.imshow('Capture - Face detection', frame)
    cv2.waitKey(0)

# Takes a single argument of the image we are trying to detect faces on
def main():
    # Read the Input Image
    frame = cv2.imread(sys.argv[1])
    # Load the strong classifier
    if not face_cascade.load(face_cascade_name):
        print('--(!)Error loading face cascade')
        exit(0)
    # Detect faces and display the result
    detectAndDisplay(frame)
    # Save result image
    cv2.imwrite("detected_"+sys.argv[1], frame)
    return 0

if __name__ == "__main__":
    main()