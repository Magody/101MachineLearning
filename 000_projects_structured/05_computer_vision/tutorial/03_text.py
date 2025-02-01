import numpy as np
import cv2 as cv
img = np.zeros((512,512,3), np.uint8)

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)