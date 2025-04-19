import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder="Data/Z"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        cropImg = img[y-offset:y+h+offset, x-offset:x+w+offset]

        cropSize = cropImg.shape        

        aspectRatio = h/w
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(cropImg, (wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wGap+wCal] = imgResize
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(cropImg, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap: hGap+hCal, :] = imgResize

        cv2.imshow("Cropped Image", cropImg)
        cv2.imshow("White Image", imgWhite)

        

    cv2.imshow("Image", img)
    # Exit if 'q' is pressed
    key = cv2.waitKey(1) #& 0xFF == ord('q')
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

# cap.release()
# cv2.destroyAllWindows()
