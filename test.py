import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time 
import tensorflow
import pyttsx3 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
offset = 20
imgSize = 300
counter = 0
labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# folder="Data/Z"
# engine = pyttsx3.init()
# engine.setProperty('rate', 100)  # Speed of speech
# last_spoken = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction,index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)
            
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(cropImg, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap: hGap+hCal, :] = imgResize
            prediction,index = classifier.getPrediction(imgWhite,draw=False)

        # current_letter = labels[index]

        # NEW: Speak only if the letter has changed
        # if current_letter != last_spoken:
        #     engine.say(current_letter)
        #     engine.runAndWait()
        #     last_spoken = current_letter

        cv2.rectangle(imgOutput, (x-offset, y-offset-50),(x-offset+90, y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput, labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("Cropped Image", cropImg)
        cv2.imshow("White Image", imgWhite)

        

    cv2.imshow("Image", imgOutput)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
