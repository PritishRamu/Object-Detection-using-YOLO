import numpy
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone  #cvzone will be used to display all the detections
import math
from sort import *




cap = cv2.VideoCapture("../Videos/people.mp4")       #For the video



model=YOLO("../Yolo-Weights/yolov8n.pt")



classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv2.imread("mask.png")    #Using .imread mask.png is read into mask.

# Tracking using SORT(Simple Online Realtime Tracker)
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limitsUp=[103,161,296,161]  #This is the value of the line after which any id crosses we count it. We have foud it using trial and error method
limitsDown=[527,489,735,489]

totalCountUp=[]
totalCountDown=[]



while True:
    success, img = cap.read()
    imRegion=cv2.bitwise_and(img,mask)   #using.bitwise_and image and mask are overlayed on each other and the resultant masked image is in imRegion
    imgGraphics=cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imgGraphics,(730,260))
    results=model(imRegion,stream=True)

    detections=np.empty((0,5))    #Using SORT we have created an empty array to keep track/count the no of people. Here detections is an empty array

    for r in results:
        boxes = r.boxes
        for box in boxes:
            box.xyxy           #Method of denoting the the dimensions of the boundary box xyxy means in x1,y1 and x2,y2 manner and another way is box.xywh where w is width and h is height.
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)     #converting all float values of x1,y1,x2,y2 to integer
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)   # here 255,0,255 is the colour of the boundary box and 3 is the thickness of the box.
                            #OR
            w,h=x2-x1,y2-y1


            #To find confidence values
            conf=(math.ceil(box.conf[0]*100))/100
            # print(conf)


            #Class Name
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass=="person":        #If it is a car then only it's confidence level and class will be displayed
                currentArray=np.array([x1,y1,x2,y2,conf])  #Passing the voundary boxes values and confidence values to the currentarray
                detections=numpy.vstack((detections,currentArray))   #Here we have used .vstack(vertical stack) to keep a track i=of all the detections by giving old detection and currentarray as parameters


    resultsTracker=tracker.update(detections)   #We have the final info in the variable resultsTracker\
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        print(result)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # converting all float values of x1,y1,x2,y2 to integer
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{int(id)}',(max(x1,0),max(y1,25)),scale=1,thickness=2,offset=6)

        #Finding Centers so that when the center crosses the line we can count it
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),radius=5,color=(0,255,0),thickness=cv2.FILLED)

        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-20<cy<limitsUp[3]+20:
            if totalCountUp.count(id)==0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255,0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[3] + 20:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    cv2.putText(img,str(len(totalCountUp)),(924,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)

    cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)


    # cv2.imshow("Masked Image Region",imRegion)
    cv2.imshow("Image",img)
    cv2.waitKey(1)         #If waitKey(1) video will run on its own if waitKey(0) then if a key is pressed only then the video will run
