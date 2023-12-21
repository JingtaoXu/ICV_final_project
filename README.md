# Car detection and speed estimation

Introduction:
The project aims to develop a car detection and speed measurement system using CCTV cameras. The system will be designed to track cars on the streets and measure their speed. The system will provide real-time data that can be used for traffic management, road safety, and law enforcement.

Objectives:
The main objectives of the project are:

1. To design an algorithm that detects cars on cctv video
2. To design a tracking method (find a way to track by plate, color, etc.)
3. To design a speed measurer and to combine it with other parts of algorithm to obtain the final product

Methodology:
The project will be divided into the following phases:

1. Phase 1: Research and Design
In this phase, we will conduct research on existing car detection and speed measurement systems. Based on the research, we will design a system that meets the requirements of our project. We will also find a training dataset for our algorithm from open source.
2. Phase 2: Prototype
In this phase, we will develop an algorithm for car detection and basic tracking
3. Phase 3: Product
Enhance prototype tracking abilities and add speed measurement
4. Phase 4: Testing and Evaluation
In this phase, we will test the system on different data. We will evaluate the performance of the algorithm.


Usage:
1. Download yolo weights with running download_yolo_weights``` sh download_yolo_weights.sh ```
   or manually download and place weights in yolo-coco from https://pjreddie.com/media/files/yolov3.weights

2. Run ```python detect.py -i video_01.mp4  -o out_video_01.avi -y yolo-coco``` 

3. Threshold and confidence for YOLO model can also be customised by -c -t arguments e.g ```python detect.py -i video_01.mp4  -o out_video_01.avi -y yolo-coco -c 0.5 -t 0.5```


Data Source:
https://data.world/datasets/cctv
https://data.world/datagov-uk/a31d6880-c159-40da-aaf6-e84396e535cb

