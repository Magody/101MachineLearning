import cv2
from WindowCapture import WindowCapture

window = WindowCapture("Explorador de archivos")

print("PRINT: ", window.list_window_names())
print("get_screen_position: ", window.get_screen_position([10, 20]))


cv2.imshow("window capture", window.get_screenshot())
cv2.waitKey(0)