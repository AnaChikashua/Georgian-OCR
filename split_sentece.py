import cv2
from imutils import contours

def split_word(file_name, image=None, save_chars=False):
    if not image:
        image = cv2.imread(f'{file_name}')

    image = cv2.imread(f'{file_name}')
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    ROI_number = 0
    ROIS = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            x, y, w, h = cv2.boundingRect(c)
            ROI = 255 - image[y:y + h, x:x + w]
            if save_chars:
                print('result/{}_{}.png'.format('xeli', ROI_number))
                cv2.imwrite('result/{}_{}.png'.format('xeli', ROI_number), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
            ROI_number += 1
            ROIS.append(ROI)
    return ROIS

if __name__ == '__main__':
    file_name = r'C:\Users\annch\OneDrive\Desktop\master\ocr\nini.jpg'
    ROIS = split_word(file_name=file_name, save_chars=True)
    # for ROI in ROIS:
    #     cv2.imshow('ROI', ROI)
    #     cv2.waitKey()
