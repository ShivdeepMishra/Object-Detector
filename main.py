# Task:
# ● Object Detection
# ● Implement an object detector which identifies the classes of the objects in
# an image or video.

# importing the opencv library and numpy library
import cv2
import numpy as np

thres = 0.45 # Threshold to detect object
nms_thresold = 0.2 # nms is used for accuracy of boxes around objects

# importing the image file
# img = cv2.imread('Image.png')

# video capture
cap = cv2.VideoCapture(1)
# parameters for defining the size of the image
cap.set(3,1280)
cap.set(4,720)
cap.set(10, 150)

# importing the class file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    # reading the class names from class file and adding into classNames list
    classNames = f.read().rstrip('\n').split('\n')

# importing configuration file
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# import weights
weightsPath = 'frozen_inference_graph.pb'

# dnn built-in method of OpenCV
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:

    # define image
    success, img = cap.read()

    # send our image to our model
    # confThreshold = 0.45, model with less than 45% surety ignores the object
    # bbx - bounding box helps in creating rectangle around the object
    # classIds helps in writing the object name
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thresold)


    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, h+y), color=(0,255,0), thickness=2)
        cv2.putText(img,classNames[classIds[i][0]-1].upper(), (box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 200, box[1] + 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # displaying the image file
    cv2.imshow("Output", img)
    cv2.waitKey(1)
