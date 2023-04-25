import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
#yolo with webcam

#creating the web cam object

cap=cv2.VideoCapture(0)
#setting the width
cap.set(3,1280)
#setting the height
cap.set(4,720)

# creating the model
model= YOLO("../Yolo_Weights/yolov8n.pt")

while True:
    success, img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0].numpy()
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,255),5)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
