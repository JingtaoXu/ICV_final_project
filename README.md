# Car detection and speed estimation


1. Clone the repo with <br>
``` git clone https://github.com/mathlabai/car-detection-and-speed-estimation.git ``` in the terminal

2. Download yolo weights with running download_yolo_weights``` sh download_yolo_weights.sh ```
   or manually download and place weights in yolo-coco from https://pjreddie.com/media/files/yolov3.weights

3. Run ```python detect.py -i video_01.mp4  -o out_video_01.avi -y yolo-coco``` 


4. Threshold and confidence for YOLO model can also be customised by -c -t arguments e.g ```python detect.py -i video_01.mp4  -o out_video_01.avi -y yolo-coco -c 0.5 -t 0.5```

