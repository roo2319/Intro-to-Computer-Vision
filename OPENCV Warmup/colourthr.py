import numpy as np
import cv2

def main():
    im = cv2.imread("mandrill.jpg")
    
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            pixelBlue = im[y,x,0]
            pixelGreen = im[y,x,1]
            pixelRed = im[y,x,2]
            if pixelBlue > 200:
                im[y,x] = (255,255,255)
            else:
                im[y,x] = (0,0,0)


    cv2.imshow("im",im)
    cv2.waitKey(0)
    cv2.imwrite("colourthr.jpg",im)

if __name__ == "__main__":
    main()