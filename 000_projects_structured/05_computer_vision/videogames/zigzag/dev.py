import numpy as np
import os
import cv2
import sys
path_os = "C:\\Users\\usuario\\Documents\\GitHub\\DataGlobal\lib\\"

sys.path.append(f"{path_os}\\computer_vision")
from utils_opencv import stackImages  # type: ignore
from WindowsCapture import WindowCapture  # type: ignore
from utils_videogame_opencv import Player  # type: ignore

class ZigZagPlayer(Player):

    def __init__(self, screen_reader, mode, verbose=3):
        super().__init__(screen_reader, mode, verbose)

    def check_white(self, img, x, y, radius, far, goingRight):

        """
        if x + far > img.shape[1] or x - far < 0:
            return True
        
        white_sum = 0
        if goingRight:
            white_sum = sum(img[y, x + far])
        else:
            white_sum = sum(img[y, x - far])

        return white_sum >= 764
        """
        # If we are out of range return true
        if x+far < img.shape[1] and x + far > 0:
            b, g, r = img[y,x+far]
            if far < 0:
                trackX, trackY = x+radius+4, y+5
                trackX2, trackY2 = x+radius+4, y-5
            else:
                trackX, trackY = x-radius-4, y+5
                trackX2, trackY2 = x-radius-4, y-5
            if trackY > 99:
                trackY = 99
            try:
                b2, g2, r2 = img[trackY, trackX]
                b3, g3, r3 = img[trackY2, trackX2]
            except IndexError:
                b2, g2, r2 = 0, 0, 0
                b3, g3, r3 = 0, 0, 0
            #print(f"R2: {r2} G2: {g2} B2: {b2}")
        else:
            #print("WHITE-OFF")
            return True

        if(sum((r,g,b)) >= 760 ) and (sum((r2,g2,b2)) < 756 and sum((r3,g3,b3)) < 756):
            return True
        elif(sum((r,g,b)) <= 237):
            return True
        
        return False

    def play(self):
        goingRight = True

        kernel = np.ones((3,3), np.uint8)

        count = 0
        offset_cut_view = 100
        pushFar = 50
        pushMiddle = 47
        pushShort = 11
        lookUp = -2

        if self.mode != Player.MODE_TEST_FROM_DATA:
            self.start()

        self.wait(0.4)

        patience = 3

        while self.game_state == Player.ON:

            count += 1
            
            if self.mode == Player.MODE_TEST_FROM_DATA:
                img = cv2.imread(os.path.join("test", self.list_images_test[2]))
            else:
                img = self.screen_reader.get_screenshot()

            if img.shape[2] >= 4:
                # remove alpha or other channels
                img = cv2.cvtColor(img, cv2.IMREAD_COLOR)

            middle_y = img.shape[0] // 2

            img = img[(middle_y - offset_cut_view):(middle_y + offset_cut_view), :, :]

            black = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Diate our make so we know it will cover edges of diamonds.
            # Then mask out our line image.
            # TODO: improve mask
            mask_diamond = cv2.inRange(hsv, np.array([153, 96, 175]), np.array([156, 255, 255]))
            mask_diamond = cv2.dilate(mask_diamond, kernel, iterations=1)
            # Gray image will be used to find circles
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray[mask_diamond > 0] = 0

            # Take our color images and find edges
            lines = cv2.Canny(img, threshold1=190, threshold2=135)
            lines[mask_diamond > 0] = 0

            # Find Houghlines and paint them on our black image array.
            # I found these parameters by using sliders in the lineDetection.py program.
            HoughLines = cv2.HoughLinesP(lines, 1, np.pi/180, threshold = 19, minLineLength = 20, maxLineGap = 1)
            if HoughLines is not None:
                for line in HoughLines:
                    coords = line[0]
                    cv2.line(black, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=18, minRadius=10, maxRadius=14)
            if circles is not None:
                circles = np.uint16(circles)

                if len(circles) > 1:
                    self.game_over()
                    continue

                for pt in circles[0, :]:
                    x, y, r = pt[0], pt[1], pt[2]
                    
                    if goingRight:
                        white_contour_sum = sum(black[y+lookUp,x+r+pushShort:x+pushMiddle])     
                    else:
                        white_contour_sum = sum(black[y+lookUp,x-pushMiddle : x-r-pushShort])

                    if goingRight:
                        cv2.line(black, (x+r+pushShort, y+lookUp), (x+pushFar, y+lookUp), [255,255,255], 3)
                    else:
                        cv2.line(black, (x-r-pushShort, y+lookUp), (x-pushFar, y+lookUp), [255,255,255], 3)
                            
                    if white_contour_sum > 0:

                        if goingRight:
                            cv2.line(img, (x+r+pushShort, y+lookUp), (x+pushFar, y+lookUp), [255,255,255], 3)
                        else:
                            cv2.line(img, (x-r-pushShort, y+lookUp), (x-pushFar, y+lookUp), [255,255,255], 3)
                                
                        
                        cv2.circle(black, (x,y), r, (255, 255, 255), 5)

                        self.log(f"{count}_{goingRight}_contour", 4)
                        self.capture_moment(f"black_{count}_{goingRight}_contour.png", black)
                        self.capture_moment(f"img_{count}_{goingRight}_contour.png", img)
                        self.action(Player.CLICK)
                        goingRight = not goingRight
                    else:
                        portion = img[y-(r):y+(r), x-(r):x+(r), :]
                        threshold_portion = np.sum(portion) // portion.size
                            
                        if goingRight:
                            answer = self.check_white(img, x, y+lookUp, r, pushFar, goingRight)
                            if not answer:
                                answer = self.check_white(img, x, y, r, pushFar, goingRight)
                        else:
                            answer = self.check_white(img, x, y+lookUp, r, -pushFar, goingRight)
                            if not answer:
                                answer = self.check_white(img, x, y, r, -pushFar, goingRight)
                        if answer:
                            if threshold_portion <= 90:
                                continue
                            print(f"{count}_{goingRight}_whitewall.png", threshold_portion)
                            cv2.circle(black, (x,y), r, (255, 255, 255), 5)
                            self.capture_moment(f"black_{count}_{goingRight}_whitewall.png", black)
                            self.capture_moment(f"img_{count}_{goingRight}_whitewall.png", img)
                            self.action(Player.CLICK)
                            goingRight = not goingRight

            else:
                if patience > 0:
                    patience -= 1
                else:
                    self.game_over()
                continue

            imgResult = cv2.bitwise_and(img, img, mask=black)

            cv2.imshow("All images", stackImages(
                    [img, gray, lines, imgResult], 2, 0.5
                )
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if goingRight:
            cv2.line(img, (x+r+pushShort, y+lookUp), (x+pushFar, y+lookUp), [255,255,255], 3)
        else:
            cv2.line(img, (x-r-pushShort, y+lookUp), (x-pushFar, y+lookUp), [255,255,255], 3)
                
        
        cv2.circle(black, (x,y), r, (255, 255, 255), 5)

        self.screenshots_moments.append((f"black_{count}_{goingRight}_final.png", black))
        self.screenshots_moments.append((f"img_{count}_{goingRight}_final.png", img))
        

print(WindowCapture.list_window_names())

window_name = "LDPlayer"
verbose = 3

screen_reader = WindowCapture(window_name)
# screen_reader.print_screen_stats()
player = ZigZagPlayer(screen_reader, Player.MODE_PRODUCTION, verbose)
player.play()