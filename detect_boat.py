import cv2
import cvzone
from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms, models
from PIL import Image

# Load the YOLO model for face detection
facemodel = YOLO('./best_2.pt')

class BoatModel(nn.Module):
    def __init__(self, num_classes):
        super(BoatModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load('resnet18-f37072fd.pth'))  # Replace with the path to your downloaded model
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Load the pretrained network
model = BoatModel(num_classes=9)
model.load_state_dict(torch.load('boat_model_res.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Capture the video frame
demo = '../boat_data/boats_video.mp4'
cap = cv2.VideoCapture(demo)

if not cap.isOpened():
    print("Error opening video.")
    exit()

while True:
    rt, frame = cap.read()
    frame = cv2.resize(frame, (720, 1080))
    if not rt:
        break

    face_results = facemodel.predict(frame, conf=0.4)
    for info in face_results:
        params = info.boxes
        for box in params:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            face = frame[y1:y1+h, x1:x1+w]

            # Resize the face image to 160x160 pixels
            face_resized = cv2.resize(face, (160, 160))
            
            # Convert the face image to a PyTorch tensor and apply the necessary transformations
            face_tensor = torch.Tensor(face_resized).permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor / 255.0 - 0.5

            # Gender and age detection
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb_resized = cv2.resize(face_rgb, (224, 224))  # Resize the face
            
            # Convert the image to PIL before applying the transformations
            face_pil = Image.fromarray(face_rgb_resized)
            face_tensor = transform(face_pil).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(face_tensor)
            
            _, predicted = torch.max(outputs, 1)
            boat_class = predicted.item()

            classes = {
                0: 'buoy',
                1: 'cruise ship',
                2: 'ferry boat',
                3: 'freight boat',
                4: 'gondola',
                5: 'inflatable boat',
                6: 'kayak',
                7: 'paper boat',
                8: 'sailboat'
            }
            
            # Draw the rectangle around the detected face and the name of the boat
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
            cv2.putText(frame, f'This boat is {classes[boat_class]}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
