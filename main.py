import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np
import argparse
import os
import csv
import datetime

# ====== Argument Parsing ======
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='detect', help='detect | segment | pose | classify')
parser.add_argument('--source', type=str, default='0', help='0=webcam or path to video/image')
args = parser.parse_args()

# ====== YOLOv11 Model Selection ======
model_files = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "pose": "yolo11n-pose.pt",
    "classify": "yolo11n-cls.pt"
}
model_file = model_files.get(args.task, "yolo11n.pt")

# ====== Load Model and Tracker ======
model = YOLO(model_file)
tracker = Sort() if args.task == 'detect' else None

# ====== Setup Output Folders ======
base_dir = "output"
image_dir = os.path.join(base_dir, "images")
video_dir = os.path.join(base_dir, "videos")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# ====== Check if Input is Image ======
is_image = args.source.lower().endswith(('.jpg', '.jpeg', '.png'))
cap = cv2.VideoCapture(0 if args.source == '0' else args.source)

# ====== Timestamp for filenames ======
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ====== Output paths ======
if is_image:
    img_path = os.path.join(image_dir, f"result_{args.task}_{timestamp}.jpg")
    csv_path = os.path.join(image_dir, f"result_{args.task}_{timestamp}.csv")
    out = None
else:
    video_path = os.path.join(video_dir, f"output_{args.task}_{timestamp}.avi")
    csv_path = os.path.join(video_dir, f"output_{args.task}_{timestamp}.csv")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

# ====== Print output info ======
if is_image:
    print(f"🖼️  Saving image to: {img_path}")
    print(f"📝 CSV log saved to: {csv_path}")
    print("🔍 Press any key to exit...")
else:
    print(f"📼 Saving video to: {video_path}")
    print(f"📝 CSV log saved to: {csv_path}")
    print("👉 Press 'q' to stop the video...")

# ====== CSV Logging Setup ======
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Class', 'Count'])

frame_count = 0

# ====== Main Loop ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, verbose=False)

    if args.task == 'classify':
        for r in results:
            label = r.names[r.probs.top1]
            conf = r.probs.top1conf
            cv2.putText(frame, f'{label} {conf:.2f}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            csv_writer.writerow([frame_count, label, 1])

    elif args.task == 'segment':
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data.cpu().numpy() if r.masks else []
            class_names = [r.names[c] for c in clss]

            for name in set(class_names):
                csv_writer.writerow([frame_count, name, class_names.count(name)])

            for i, mask in enumerate(masks):
                color_mask = (mask * 255).astype(np.uint8)
                color_mask = cv2.merge([color_mask]*3)
                frame = cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0)

                x1, y1, x2, y2 = map(int, boxes[i])
                label = r.names[clss[i]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif args.task == 'pose':
        for r in results:
            if r.keypoints is None:
                continue
            keypoints = r.keypoints.xy.cpu().numpy()
            scores = r.keypoints.conf.cpu().numpy()

            for person_kps, confs in zip(keypoints, scores):
                for (x, y), c in zip(person_kps, confs):
                    if c > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                            (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
                for i, j in skeleton:
                    if confs[i] > 0.5 and confs[j] > 0.5:
                        pt1 = tuple(map(int, person_kps[i]))
                        pt2 = tuple(map(int, person_kps[j]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

            csv_writer.writerow([frame_count, 'person_pose', len(keypoints)])

    else:  # detect + SORT tracking
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            dets = []

            class_names = [r.names[c] for c in clss]
            for name in set(class_names):
                csv_writer.writerow([frame_count, name, class_names.count(name)])

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = r.names[cls]
                dets.append([x1, y1, x2, y2, conf])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if tracker:
                dets_np = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
                tracks = tracker.update(dets_np)
                for tr in tracks:
                    x1, y1, x2, y2, track_id = tr.astype(int)
                    cv2.putText(frame, f'ID: {track_id}', (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ====== Save Output ======
    if out:
        out.write(frame)

    cv2.imshow(f"YOLO11 - {args.task}", frame)

    if is_image:
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved image to: {img_path}")
        print(f"✅ Saved CSV to: {csv_path}")
        print("🔍 Press any key to exit...")
        cv2.waitKey(0)
        break

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Cleanup ======
cap.release()
if out:
    out.release()
csv_file.close()
cv2.destroyAllWindows()
