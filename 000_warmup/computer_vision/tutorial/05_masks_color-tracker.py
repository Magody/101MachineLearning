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

# LOAD AN IMAGE USING 'IMREAD'

h_min = 99
s_min = 4
v_min = 27
h_max = 170
s_max = 255
v_max = 255

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",h_min,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",h_max,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",s_min,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",s_max,255,empty)
cv2.createTrackbar("Val Min","TrackBars",v_min,255,empty)
cv2.createTrackbar("Val Max","TrackBars",v_max,255,empty)

while True:
    img = cv2.imread(f"{path_data}/lena.png")  # BGR
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(f"{h_min},{s_min},{v_min},{h_max},{s_max},{v_max}")
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    # Check if HSV image lies in this range to detect color
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    imgStack = stackImages(0.6, [img,imgHSV,mask,imgResult])
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)
