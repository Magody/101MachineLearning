import win32api as wapi
import pyautogui
import shutil
import os
from time import sleep
import cv2

pyautogui.PAUSE = 0

class Player:
    
    CLICK = 0

    ON = 1
    OFF = 0

    MODE_TEST_FROM_DATA = 0
    MODE_DEVELOP = 1
    MODE_PRODUCTION = 2

    def __init__(self, screen_reader, mode, verbose):
        self.game_state = Player.ON
        self.mode = mode

        self.screenshots_moments = []

        path_images = os.path.join("images")
        shutil.rmtree(path_images)

        if not os.path.isdir(path_images):
            os.makedirs(path_images)

        self.list_images_test = []

        self.screen_reader = screen_reader
        self.verbose = verbose
        self.key_list = ["\b"]
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
            self.key_list.append(char)

        
        if self.mode == Player.MODE_TEST_FROM_DATA:
            list_test = filter(lambda f: "png" in f, os.listdir(os.path.join("test")))
            self.list_images_test = list(list_test)
            self.log(f"List of images: {self.list_images_test}", 4)

    def log(self, message, verbose_minimum=1):
        if self.verbose >= verbose_minimum:
            print(message)

    def wait(self, time):
        sleep(time)

    def start(self):
        for i in range(3, 0, -1):
            self.log(f"Starting in {i}...")
            self.wait(1)
        
        self.action(Player.CLICK)

    def get_input_keys(self):
        keys = []
        for key in self.key_list:
            if wapi.GetAsyncKeyState(ord(key)):
                keys.append(key)
        return keys
    
    def capture_moment(self, filename, img):
        if self.mode == Player.MODE_DEVELOP:
            cv2.imwrite(os.path.join("images", filename), img)
        elif self.mode == Player.MODE_PRODUCTION:
            self.screenshots_moments.append((filename, img))
            if len(self.screenshots_moments) > 30:
                self.screenshots_moments.pop(0)

    def action(self, action_key : int):
        if action_key == Player.CLICK:
            pyautogui.click()
            self.wait(0.01)
            # self.log("CLICK", 4)

    def play(self):
        print("PLAY NOT IMPLEMENTED")

    def game_over(self):
        self.game_state = Player.OFF
        self.log("GAME OVER")

    def __del__(self):
        print("SAVING BEST MOMENTS...")

        for moment in self.screenshots_moments:
            filename = moment[0]
            img = moment[1]
            cv2.imwrite(os.path.join("images", filename), img)