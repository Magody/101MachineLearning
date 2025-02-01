import cv2
import sys
sys.path.append("/home/magody/programming/python/data_science/lib/computer_vision")
from HandTracking import HandDetector  # type: ignore
from utils_opencv import cornerRect  # type: ignore
import numpy as np


cap = cv2.VideoCapture(2)
cap.set(3, 800)
cap.set(4, 400)
detector = HandDetector(min_detection_confidence=0.8)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.find_hands(img, flipType=True)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        if lmList1:

            # send img and return image to draw distance
            point_index = lmList1[8][0:2]
            point_middle = lmList1[12][0:2]
            l, _, _ = detector.find_distance(point_index, point_middle, img=None)
            print(l)
            if l < 40:
                cursor = lmList1[8][0:2]  # index finger tip landmark
                # call the update here
                for rect in rectList:
                    rect.update(cursor)

    
    ## Draw Transperency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)


    out = img.copy()
    mask = imgNew.astype(bool)
    alpha = 0.5
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)