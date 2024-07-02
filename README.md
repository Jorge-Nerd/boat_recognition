# boat_recognition
# Boat Classification and Face Detection

This project demonstrates a pipeline for detecting faces in video frames using YOLOv8 and classifying boats using a model in PyTorch.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Files](#model-files)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

To run this project, you need to install the required dependencies. Use the following commands to install them:

````
bash
pip install opencv-python opencv-python-headless
pip install cvzone
pip install ultralytics
pip install torch torchvision
pip install numpy
pip install pillow
````

## Usage
To run the face detection and boat classification, use the following command:

````
python detect_boat.py
````

## Model Files
- best_2.pt: Pretrained YOLO model for boat detection.
- resnet18-f37072fd.pth: Pretrained ResNet18 model.
- boat_model_res.pth: Pretrained model for boat classification.


![image](https://github.com/Jorge-Nerd/boat_recognition/assets/77848827/7c7eb870-595a-44b0-9c58-4fae9c594cce)

