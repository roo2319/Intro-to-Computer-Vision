import numpy as np
import cv2

def main():
    im = np.zeros((256,256,3),np.uint8)
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            im[y,x,0] = x
            im[y,x,1] = y
            im[y,x,2] = 255 - im[y,x,1]

    cv2.imshow("im",im)
    cv2.waitKey(0)
    cv2.imwrite("rainbow.jpg",im)

if __name__ == "__main__":
    main()