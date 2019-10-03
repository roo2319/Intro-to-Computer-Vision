import numpy as np
import cv2

def main():
    im = cv2.imread("mandrill.jpg")
    
    gray_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            pixel = gray_im[y,x]
            gray_im[y,x] = 255 if pixel > 128 else 0

    cv2.imshow("im",gray_im)
    cv2.waitKey(0)
    cv2.imwrite("mandrillThreshold.jpg",gray_im)

if __name__ == "__main__":
    main()