import numpy as np
import cv2

def main():
    im = cv2.imread("mandrill.jpg")
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,4)

    cv2.imshow("im",im)
    cv2.waitKey(0)
    cv2.imwrite("adaptthr.jpg",im)

if __name__ == "__main__":
    main()