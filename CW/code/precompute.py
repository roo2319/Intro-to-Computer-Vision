import ellipses
import lineswithgradient
import cv2
import os
import pickle
import helpers

test = os.listdir("../test_images")
generated = os.listdir("../houghs")

for image in test:
    image=os.path.splitext(image)[0]
    print(image)
    if "lines_" + image not in generated:
        im = cv2.imread("../test_images/"+image+".jpg")
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        houghSpace = lineswithgradient.hough(im)
        with open("../houghs/lines_"+image,'wb') as f:
            pickle.dump(houghSpace,f)
    
    if "ellipses_" + image not in generated:
        im = cv2.imread("../test_images/"+image+".jpg")
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        mag = helpers.sobel(im)[0]
        houghSpace = ellipses.makeHough(mag)
        with open("../houghs/ellipses_"+image,'wb') as f:
            pickle.dump(houghSpace,f)
    