import cv2
import numpy as np

path_data = "C:\\Users\\usuario\\Documents\\GitHub\\DataGlobal\\data\\images_opencv"

def empty(a):
   pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

width = 512
height = 512
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
FILLED = -1

img_rectangle = cv2.rectangle(200 * np.ones([width, height, 3], np.uint8), (100, 100), (400, 400), WHITE, FILLED)
img_circle = cv2.circle(100 * np.ones([width, height, 3], np.uint8), (250, 250), 170, WHITE, FILLED)

# A mask must be 2D
img_circle_mask = cv2.circle(np.zeros([width, height], np.uint8), (250, 250), 120, WHITE, FILLED)

img_rectangle_green = cv2.rectangle(200 * np.ones([width, height, 3], np.uint8), (100, 100), (400, 400), GREEN, FILLED)
img_and = cv2.bitwise_and(img_rectangle, img_circle)
img_and_cut = cv2.bitwise_and(img_rectangle_green, img_rectangle_green, mask=img_circle_mask)

# AND - "sums" values
# OR - one true of both or both true
# XOR = one true but not both
# NOT - complement of on and off
# AND with MASK - CUTOFF
# https://omes-va.com/operadores-bitwise/

img_or = cv2.bitwise_or(img_rectangle, img_circle)
img_xor = cv2.bitwise_xor(img_rectangle, img_circle)
img_not = cv2.bitwise_not(img_rectangle)

imgStack = stackImages(0.6, [
    [img_rectangle, img_circle, img_and],
    [img_rectangle_green, img_circle_mask, img_and_cut],
    [img_or, img_xor, img_not]
    ])
cv2.imshow("Stacked Images", imgStack)
cv2.waitKey(0)
