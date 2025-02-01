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


img = cv2.imread(f"{path_data}/lena.png")

width = 512
height = 512
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
FILLED = -1
# formula: https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html

img_rectangle = cv2.rectangle(200 * np.ones([width, height, 3], np.uint8), (100, 100), (400, 400), WHITE, FILLED)
img_circle = cv2.circle(100 * np.ones([width, height, 3], np.uint8), (250, 250), 170, WHITE, FILLED)
img_circle_mask = cv2.circle(np.zeros([width, height], np.uint8), (250, 250), 120, WHITE, FILLED)
img_rectangle_green = cv2.rectangle(200 * np.ones([width, height, 3], np.uint8), (100, 100), (400, 400), GREEN, FILLED)

img_addition = cv2.add(img, img_rectangle_green)
img_addition_mask = cv2.add(img, img_rectangle_green, mask=img_circle_mask)

img_addition_weighted = cv2.addWeighted(img, 1, img_rectangle_green, 0.3, 0)

imgStack = stackImages(0.6, [
    [img, img_rectangle_green],
    [img_addition, img_addition_mask],
    [img, img_addition_weighted],
    ])
cv2.imshow("Stacked Images", imgStack)
cv2.waitKey(0)
