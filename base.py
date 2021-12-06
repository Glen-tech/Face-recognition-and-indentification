# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 19:50:32 2021

Based on tutorial https://www.youtube.com/watch?v=PmZ29Vta7Vc

@author: glen_
"""

import numpy as np
import cv2 
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

lables= {}
with open("labels.picle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


ESCAPE = 27
ENTER = 13

cam = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cam.read()
    to_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(to_gray, scaleFactor= 1.5 , minNeighbors = 5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = to_gray[y:y+h, x:x+w] # region of interest 
        roi_color = frame[y:y+h, x:x+w]
        
        # recognizer in deep learned model
        
        id_,conf= recognizer.predict(roi_gray) # label en conf
        if conf >= 40 and conf <= 50:
            print("conf:",conf)
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            
          
         
        img_item = "my-image.png"
        img_item_color = "my-image-color.png"
        
        
        cv2.imwrite(img_item,roi_gray)
        cv2.imwrite(img_item_color, roi_color)
        
        color = (255, 0, 0) # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y) , color , stroke)
        cv2.putText(frame, labels[id_], (x,y), font , 2 , color=(255, 255, 255))
            
        
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        
        #for(ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh),color , stroke)
        
        #smile = smile_cascade.detectMultiScale(roi_gray)
        
        #for(ex,ey,ew,eh) in smile:
        #    cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh),color , stroke)
            
            
    cv2.imshow('webcam',frame)
    
    key = cv2.waitKey(1)
    if (key == ENTER) or (key == ESCAPE): # enter key 
        break
    
    
    
cam.release()
cv2.destroyAllWindows()    
