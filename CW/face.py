from __future__ import print_function
import cv2
import argparse
import matplotlib.pyplot as plt
import sys
def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    haar=cv2.CASCADE_SCALE_IMAGE
    faces = face_cascade.detectMultiScale(frame_gray,1.1,1,0|haar, (50, 50), (500,500))
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
    cv2.imshow('Capture - Face detection', frame)
    cv2.waitKey(0)
cv2.samples.addSamplesDataSearchPath("C:/Users/Theano/Desktop/Into-to-Computer-Vision/CW")
face_cascade_name = "frontalface.xml"
face_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
frame=cv2.imread(sys.argv[1])
detectAndDisplay(frame)