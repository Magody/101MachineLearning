import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/magody/programming/python/data_science/lib/computer_vision")
from utils_opencv import stackImages  # type: ignore
 
path_haarcascades = "/home/magody/programming/python/data_science/data/haarcascades"
import cv2
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(4, 800)
cap.set(10,20)
while True:
    success, img = cap.read()
    # img = imutils.resize(img, width=900)
    img = cv2.flip(img, 1)
    
    faceCascade= cv2.CascadeClassifier(f"{path_haarcascades}/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(img,0,200)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(imgCanny,(x,y),(x+w,y+h),(255,0,0),2)
    
    imgStack = stackImages([img,imgGray,imgCanny], 3, 0.6)
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)