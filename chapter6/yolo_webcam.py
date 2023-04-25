import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import math
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
            #bounding box
            x1,y1,x2,y2=box.xyxy[0].numpy()
            #x1,y1,w,h=box.xywh[0].numpy()
            w,h=x2-x1,y2-y1
            bbox=int(x1),int(y1),int(w),int(h)

            conf=math.ceil(box.conf[0]*100)/100
            print("the confidence is",conf)

            cls=box.cls[0]

            cvzone.cornerRect(img,bbox)
            cvzone.putTextRect(img,f'{cls},{conf}',(max(0,int(x1)),max(35,int(y1))),scale=0.7,thickness=0.5)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        break



   # cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
