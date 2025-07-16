## CodeAlpha_ObjectDetectionAndTracking
Real-time object detection and tracking using YOLOv11 and SORT. Processes webcam or video input with OpenCV, detects objects frame-by-frame, applies tracking with unique IDs, and displays results with labels in real time.

### 🔍 Real-Time Object Detection, Segmentation, Pose Estimation & Tracking
This project demonstrates a real-time computer vision system that performs **object detection**, **instance segmentation**, **pose estimation**, **object classification**, and **object tracking** using **YOLOv11** and **SORT**. It supports live webcam feeds and video file inputs, with output saved as video, image frames, and CSV logs.

### 📌 Project Objectives
- Capture real-time video input from **webcam and video file** using **OpenCV**.
- Perform object detection with pre-trained **YOLOv11** models.
- Support **additional vision** tasks: segmentation, pose estimation, and classification.
- Track objects across frames using the **SORT algorithm**.
- Display live outputs with bounding boxes, labels, and tracking IDs.
- Save results (videos, images, and logs) for further analysis.

### Technologies Used
Python 3.x          - Programming language                         
OpenCV              - Video processing and real-time rendering     
Ultralytics YOLOv11 - Object detection, segmentation, etc.         
SORT                - Object tracking                              
NumPy               - Numerical operations                         
Anaconda            - Virtual environment setup (recommended) 

 **Download YOLOv11 Models:**  
- [Ultralytics YOLOv11 Release Page](https://github.com/ultralytics/assets/releases)

**Final comments to run the project**
python main.py --task detect --source persons.jpg      -- detection on image
python main.py --task detect --source 0                -- **detection on webcam**
python main.py --task segment --source persons.jpg     -- segmentation
python main.py --task pose --source persons.jpg        -- pose
python main.py --task classify --source car.jpg        -- classification
python main.py --task detect --source video.mp4        -- detection on video

**Learnings & Outcomes**
Gained practical experience in integrating deep learning models in real-time systems.
Understood how to work with live video streams using OpenCV.
Implemented multi-task vision pipelines (detection, segmentation, pose, classification).
Learned how to track objects using SORT and record results.

✅ Annotated output videos, frames, and tracking logs are available in the results/ directory for demonstration purposes.

Below are three sample outputs; additional results can be found in the results/ folder.:

<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/ed57f278-f536-4cfe-9966-7e281e65ff72" />
<img width="319" height="239" alt="image" src="https://github.com/user-attachments/assets/5b27024e-fc32-4ea6-bcf0-75a2f1330295" />
<img width="318" height="214" alt="image" src="https://github.com/user-attachments/assets/7e7c0c3e-c310-4741-8247-8b3ac8f7aa9f" />





