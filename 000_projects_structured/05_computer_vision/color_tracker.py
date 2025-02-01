import cv2
import sys
import numpy as np

path_os = "C:\\Users\\usuario\\Documents\\GitHub\\DataGlobal\lib\\"

sys.path.append(f"{path_os}\\computer_vision")
from utils_opencv import stackImages  # type: ignore

def empty(a):
   pass

h_min = 72
s_min = 63
v_min = 194
h_max = 87
s_max = 135
v_max = 255

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",h_min,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",h_max,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",s_min,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",s_max,255,empty)
cv2.createTrackbar("Val Min","TrackBars",v_min,255,empty)
cv2.createTrackbar("Val Max","TrackBars",v_max,255,empty)

cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        continue
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
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)
    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)
    imgStack = stackImages([img,imgHSV,mask,imgResult], 2, 0.6)
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)