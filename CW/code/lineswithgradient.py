import numpy as np
import cv2
import math
from helpers import sobel


def hough(im):

    height, width = im.shape
    diagonal = int(np.ceil(np.sqrt(width*width + height*height)))
    houghSpace = np.zeros((2*diagonal, 360))

    mag, ang = sobel(im)
    # position in hough space represents location in these lists
    p = range(-diagonal, diagonal)

    for y in range(height):
        for x in range(width):
            if mag[y, x] != 0:
                t = [angleD+ang[y][x] for angleD in np.arange(-0.1, 0.1, 0.02)]
                for t_index in range(len(t)):
                    angle = t[t_index]
                    # As range from -diagonal to diagonal,
                    # Add diagonal so range goes from 0 to 2 * diagonal
                    p_index = int(x * np.cos(angle) + y *
                                  np.sin(angle)) + diagonal
                    houghSpace[p_index, int(math.degrees(angle)) % 360] += 1

    angles = []
    for p_index in range(houghSpace.shape[0]):
        for t_index in range(houghSpace.shape[1]):
            # Theshold
            if houghSpace[p_index, t_index] >= 15:
                angles.append(t_index)
                # Unused code to draw lines
                # angle = math.radians(t_index)
                # distance = p_index-diagonal

                # a = np.cos(angle)
                # b = np.sin(angle)
                # # Find base coordinates
                # x0 = np.cos(angle) * distance
                # y0 = np.sin(angle) * distance

                # # Generate endpoints, to create a long line
                # x1 = int(x0 + 1000*(-b))
                # y1 = int(y0 + 1000*(a))
                # x2 = int(x0 - 1000*(-b))
                # y2 = int(y0 - 1000*(a))
                # # cv2.line(im, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return set(map(lambda x: x//10, angles))


def findLines(image):
    return hough(image)


def main():
    image = cv2.imread('../test_images/dart2.jpg')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)
    findLines(frame_gray)


if __name__ == "__main__":
    main()
