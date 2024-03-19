import torch
import torchvision
import cv2
import numpy as np

# Model init
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

# Check whether GPU is usable or not 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define Labels
class_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'sheep',
    'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Open Video
video_path = "video_sample3.mp4"
cap = cv2.VideoCapture(video_path)

# Take Resolution Infos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Wr init
out = cv2.VideoWriter('maskrcnn_resnet50_out_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (frame_width, frame_height))

# Video Process Loop
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Converting video frame to tensor format
        input_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)

        # Object detection and results
        with torch.no_grad():
            prediction = model(input_tensor)[0]

        # Applying results to the frames
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > 0.5: # threshold
                start_x, start_y, end_x, end_y = box.cpu().numpy().astype("int")
                class_name = class_names[label]
                confidence = score.item()  # confidance
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Show and save the output image
        cv2.imshow('Object Detection', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()
