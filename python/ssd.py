import numpy as np
import argparse
import cv2
import sys

parser = argparse.ArgumentParser(
        description="Script to run MobileNet-SSD object detection network")
parser.add_argument("--prototxt", default="ssd/MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="ssd/MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

classNames  = open("ssd/tensorflow/coco.names").read().strip().split("\n")

# net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
net = cv2.dnn.readNetFromTensorflow("ssd/tensorflow/frozen_inference_graph.pb", "ssd/tensorflow/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# LABELS = open("ssd/tensorflow/coco.names").read().strip().split("\n")

np.random.seed(67)
COLORS = np.random.randint(0, 255, size=(len(classNames), 3), dtype="uint8")

print("Starting camera")
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    sys.exit()

print("Starting detect objects")

while True:

    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    blob = cv2.dnn.blobFromImage(frame_resized,size=(300, 300),
        swapRB=True, crop=False)

    net.setInput(blob)
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction

        if confidence > args.thr: # Filter prediction
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)

            print(xLeftBottom)

            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0
            widthFactor = frame.shape[1]/300.0

            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            # Draw location of object
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))


            # print("Class:", classNames[class_id])
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0],
                                         yLeftBottom + baseline),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break
