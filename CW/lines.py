import cv2
import numpy as np


def convolute(im, kernel):
    newim = np.zeros(im.shape)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            val = 0
            for convY in range(-1, kernel.shape[0]-1):
                for convX in range(-1, kernel.shape[1]-1):
                    try:
                        val += (im[y-convY, x-convX] *
                                kernel[convY+1, convX+1])
                    except:
                        pass
            newim[y, x] = val
    return newim


def magnitude(x: int, y: int):
    return np.float32(np.sqrt(np.square(x) + np.square(y)))


def angle(x, y):
    return np.float32(np.arctan2(y, x))


def sobel(im):
    sobelX = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    dx = convolute(im, sobelX)
    # cv2.imshow("edge", dx)
    # cv2.waitKey(0)
    sobelY = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    dy = convolute(im, sobelY)
    # cv2.imshow("edge", dy)
    # cv2.waitKey(0)
    mag = magnitude(dx, dy)
    _, mag = cv2.threshold(mag, 1, 255, cv2.THRESH_BINARY)
    ang = angle(dx, dy)
    return mag, ang
    # cv2.imshow("gradient",mag)
    # cv2.waitKey(0)
    # cv2.imshow("ang",ang)
    # cv2.waitKey(0)


def hough(im, angles, origim):
    # Distance, angle
    width, height = im.shape
    diagonal = int(np.ceil(np.sqrt(width*width + height*height)))
    houghSpace = np.zeros((2*diagonal, angles))

    mag, ang = sobel(im)

    p = range(-diagonal, diagonal)
    t = [x*np.pi/angles for x in range(angles)]

    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if mag[y, x] != 0:
                for t_index in range(len(t)):
                    angle = t[t_index]
                    # As range from -diagonal to diagonal,
                    # Add diagonal so range goes from 0 to 2 * diagonal
                    p_index = int(x * np.cos(angle) + y *
                                  np.sin(angle)) + diagonal
                    houghSpace[p_index, t_index] += 1

    print("Thresholding")
    for p_index in range(houghSpace.shape[0]):
        for t_index in range(houghSpace.shape[1]):
            if houghSpace[p_index, t_index] < 100:
                houghSpace[p_index, t_index] = 0
    cv2.imshow("HOG", houghSpace)
    cv2.waitKey(0)

    print("Overlaying")
    for p_index in range(houghSpace.shape[0]):
        for t_index in range(houghSpace.shape[1]):
            if houghSpace[p_index, t_index] > 0:
                angle = t[t_index]
                distance = p[p_index]
                a = np.cos(angle)
                b = np.sin(angle)
                x0 = np.cos(angle) * distance
                y0 = np.sin(angle) * distance
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(origim, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("overlay", origim)
    cv2.waitKey(0)


def main():
    origim = cv2.imread("line.png")
    im = cv2.GaussianBlur(cv2.cvtColor(origim, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    houghSpace = hough(im, 16, origim)


if __name__ == "__main__":
    main()
