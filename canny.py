import cv2
from imutils import contours

image = cv2.imread('data/Screenshot 2023-06-30 131823.png')
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
canny = cv2.Canny(blur, 120, 255, 1)

cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts, _ = contours.sort_contours(cnts, method="left-to-right")
# cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")
cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[0], cv2.boundingRect(c)[1]))


min_area = 100
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1
cv2.imshow('image', image)
cv2.waitKey()