import cv2

path_data = "C:\\Users\\usuario\\Documents\\GitHub\\DataGlobal\\data\\images_opencv"

# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread(f"{path_data}/lena.png")  # BGR

img[:, :, 2] = 127  # Modifying red to half

# Regions of image
section = img[280:340, 330:390]
img[173:233, 0:60] = section

# DISPLAY
cv2.imshow("Lena Soderberg",img[:, :, :])

print("type: ", type(img))
print("shape: ", img.shape)
print("img min: ", img.min())
print("img max: ", img.max())

# wait for key forever
cv2.waitKey(0)

