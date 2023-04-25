from ultralytics import YOLO
import cv2


model = YOLO('../Yolo_Weights/yolov8n.pt')

results=model("Images/parking.png", show =True)
cv2.waitKey(0)
