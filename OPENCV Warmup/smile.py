import cv2
import numpy as np

def main():
    im = np.zeros((256,256,3), (np.uint8))
    im[:,:] = (0,0,255) #B,G,R WTF
    #Create and underline text
    cv2.putText(im,"HelloOpenCV!",(70,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),1)
    cv2.line(im,(74,90),(190,90),(255,0,0),2)
    #Now make an ellipse
    cv2.ellipse(im,(130,180),(25,25),180,180,360,(0,255,0),2)
    cv2.circle(im,(130,180),50,(0,255,0),2)
    cv2.circle(im,(110,160),5,(0,255,0),2)
    cv2.circle(im,(150,160),5,(0,255,0),2)
    cv2.imshow('im',im)
    cv2.waitKey(0)
    cv2.imwrite('smile.jpeg',im)
    return 0


if __name__ == "__main__":
    main()