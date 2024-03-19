import cv2
import numpy as np

# Path and configuration of the model
model_path = "yolov3.weights"
config_path = "yolov3.cfg"
labels_path = "coco.names"  # Path to labels file

# Read the labels
with open(labels_path, "r") as f:
    classes = f.read().strip().split("\n")

# YOLO init
net = cv2.dnn.readNet(model_path, config_path)

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open Video
cap = cv2.VideoCapture('video_sample1.mp4')

# Wr init
fourcc = cv2.VideoWriter_fourcc(*'XVID')
_out = cv2.VideoWriter('yolov3_out_1.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Setting NMS parameters
confThreshold = 0.5  
nmsThreshold = 0.4   

# Video Process Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Object detection with YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detected objects
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:  # Confidence threshold check
                # object detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Drawing detected objects
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write to output video
    _out.write(frame)

    # Show the results
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release
cap.release()
_out.release()
cv2.destroyAllWindows()
