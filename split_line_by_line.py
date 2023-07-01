import cv2
import numpy as np
img = cv2.imread('data\Screenshot 2023-06-30 131823.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# create black background of same image shape
black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

# find contours from threshold image
contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# draw contours whose area is above certain value
area_threshold = 7
for c in contours:
    area = cv2.contourArea(c)
    if area > area_threshold:
        black = cv2.drawContours(black,[c],0,(255,255,255),2)
        cv2.imshow('',black)
        cv2.waitKey()