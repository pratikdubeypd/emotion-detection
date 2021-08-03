import sys
import os
import cv2 , time
import pandas as pd
import numpy as np

face_recognizer = cv2.face.FisherFaceRecognizer_create()
faceDet = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt_tree.xml")
emotions = ["Angry","Disgust","Fear", "Happy", "Sad","Surprise", "Neutral"]

cap=cv2.VideoCapture(0)
face_recognizer.read("trained.yml")

while(True):
    ret , frame = cap.read()
    # frame=cv2.imread("20200420_212646.jpg")
    # frame=cv2.resize(frame,(250,250))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
        out = cv2.resize(gray, (48, 48)) #Resize face so all images have same size
        pred, conf = face_recognizer.predict(out)

        color=(255,0,)
        stroke=2
        end_cord_x=x + w
        end_cord_y=y + h
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color ,stroke)
        if conf>=0:
            font=cv2.FONT_HERSHEY_COMPLEX
            name=emotions[pred]
            color=(255,255,0)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
    time.sleep(0.5)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break