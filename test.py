import cv2
import numpy as np
import utlis

path = '1.jpg'

img = cv2.imread(path)

img = cv2.resize(img, (500, 700))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 100, 110)

imageArray = ([img, imgGray, imgBlur, imgCanny])

imgStack = utlis.stackImages(imageArray, 0.5)

cv2.imshow('Sample Display', imgStack)
cv2.waitKey(0)