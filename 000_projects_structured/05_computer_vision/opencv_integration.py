import cv2
import imutils
import sys
sys.path.append("/home/magody/programming/python/data_science/lib/computer_vision")
from HandTracking import HandDetector  # type: ignore
from utils_opencv import cornerRect, stackImages  # type: ignore
import numpy as np

path_haarcascades = "/home/magody/programming/python/data_science/data/haarcascades"

cascade_face= cv2.CascadeClassifier(f"{path_haarcascades}/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(4, 800)
cap.set(10,20)
detector = HandDetector(min_detection_confidence=0.8)
colorR = (255, 255, 100)

cx, cy, w, h = 100, 100, 150, 150

myColors = [[61,71,76,85,255,238]]
## BGR colors
myColorValues = [[0, 255, 0]]
myPoints =  []  ## [x , y , colorId ]

def findColor(img,myColors,myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints=[]
    for color in myColors:
        lower = np.array(color[0:3]) # mins hue,sat,value...
        upper = np.array(color[3:6]) # max hue,sat,value..
        mask = cv2.inRange(imgHSV,lower,upper)
        x,y=getContours(mask)
        cv2.circle(img,(x,y),15,myColorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count +=1
        #cv2.imshow(str(color[0]),mask)
    return newPoints


def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2,y

def drawOnCanvas(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(img, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

class DragRect():
    def __init__(self, posCenter, size=[100, 100]):
        self.posCenter = posCenter
        self.size = size
        
    def is_point_inside(self, point):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        # Horizontal
        is_inside_horizontal = (cx - w // 2) < point[0] and point[0] < cx + (w // 2)
        # Vertical
        is_inside_vertical = cy - (h // 2) < point[1] and point[1] < cy + (h//2)        
        inside = is_inside_horizontal and is_inside_vertical
        return inside
                         

    def update(self, cursor):
        if self.is_point_inside(cursor):
            self.posCenter = cursor

counter = 3
rectList = []
for i in range(2):
    for j in range(2):
        rectList.append(DragRect([j * 160 + 200, i * 160 + 50], size=[120, 120]))
        counter -= 1
        if counter <= 0:
            break


while True:
    success, img = cap.read()
    # img = imutils.resize(img, width=900)
    img = cv2.flip(img, 1)
    hands, img = detector.find_hands(img, flipType=True)

    p1_enable = [0,0]
    p2_enable = [0,0]
    if hands:
        # Hand 1
        for i_hand,hand in enumerate(hands):
        
            lmList1 = hand["lmList"]  # List of 21 Landmark points
            if lmList1:

                # send img and return image to draw distance
                point_index = lmList1[8][0:2]
                point_middle = lmList1[12][0:2]
                if i_hand == 0:
                    p1_enable = point_index
                else:
                    p2_enable = point_index
                l, _, _ = detector.find_distance(point_index, point_middle, img=None)
                # print(l)
                if l < 50:
                    cursor = lmList1[8][0:2]  # index finger tip landmark
                    # call the update here
                    # print("IS LESS", cursor)
                    for rect in rectList:
                        rect.update(cursor)

    enable_draw = False
    ## Draw Transperency
    imgNew = np.zeros_like(img, np.uint8)
    for i_rect, rect in enumerate(rectList):
        cx, cy = rect.posCenter
        w, h = rect.size
        cr = colorR
        if i_rect == 0:
            if rect.is_point_inside(p1_enable) or rect.is_point_inside(p2_enable):
                enable_draw = True
                cr = (0,255,0)
            else:
                cr = (255,0,0)
            
            cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), cr, cv2.FILLED)

    # faces = cascade_face.detectMultiScale(img,1.1,4)
 
    # for (x,y,w,h) in faces:
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    if enable_draw:
        newPoints = findColor(img, myColors,myColorValues)
        if len(newPoints)!=0:
            for newP in newPoints:
                myPoints.append(newP)
        if len(myPoints)!=0:
            drawOnCanvas(myPoints,myColorValues)

    out = img.copy()
    mask = imgNew.astype(bool)
    alpha = 0.5
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)