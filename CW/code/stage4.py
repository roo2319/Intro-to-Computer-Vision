import numpy as np
import cv2

def classify(image):
    _,thr = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr= cv2.dilate(thr,np.ones((5,5)),iterations=1)
    thr = cv2.erode(thr,np.ones((5,5)),iterations=1)

    cv2.imshow("thr",thr)
    cv2.waitKey(0)
    return True
