import cv2
import numpy as np
import utlis

path = '1.jpg'
img = cv2.imread(path)

# Preprocess
img = cv2.resize(img, (400, 500))
imgContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 100, 110) 

cimg = cv2.cvtColor(imgBlur, cv2.COLOR_GRAY2BGR)

# Find all contours
contours, hierachy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

circles = cv2.HoughCircles(imgBlur, cv2.HOUGH_GRADIENT, 1, 20, param1 = 500, param2 = 40, minRadius = 0, maxRadius = 0)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)



imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBlank, imgBlank, imgBlank])
imgStack = utlis.stackImages(imageArray, 0.5)

cv2.imshow('dd', imgStack)
cv2.imshow('Sample Display', cimg)
cv2.waitKey(0)

# plaque counting