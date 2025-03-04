import cv2
import torch
import numpy as np
from sort import Sort
import mediapipe as mp

# YOLOv5 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# SORT 트래커 초기화 (Multi-Object Tracking)
tracker = Sort()

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 비디오 파일 경로
video_path = r"C:\Users\tmdgu\Downloads\msqt.mp4"
cap = cv2.VideoCapture(video_path)

# 사람별 스쿼트 기록 (ID: 횟수)
squat_counts = {}
prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5로 사람 감지
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
    
    # 사람(class 0)만 필터링
    person_detections = [det[:4] for det in detections if int(det[5]) == 0]
    
    # SORT 트래커로 ID 할당
    tracked_objects = tracker.update(np.array(person_detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)

        # ROI(관심 영역) 추출하여 Pose Estimation 적용
        person_crop = frame[y1:y2, x1:x2]

        # 크롭된 이미지가 비어 있지 않다면, Pose Estimation 진행
        if person_crop.size > 0:
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = pose.process(person_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 관절 위치 추출
                hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y

                # ID별 스쿼트 카운트 관리
                if obj_id not in squat_counts:
                    squat_counts[obj_id] = 0
                    prev_positions[obj_id] = "up"

                # 스쿼트 감지
                if hip_y > knee_y:
                    if prev_positions[obj_id] == "up":
                        squat_counts[obj_id] += 1
                        print(f"Person {obj_id} - Squat count: {squat_counts[obj_id]}")
                    prev_positions[obj_id] = "down"
                else:
                    prev_positions[obj_id] = "up"

                # Bounding Box & ID 출력
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {obj_id}: {squat_counts[obj_id]}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print(f"Person {obj_id} cropped area is empty!")

    # 화면 출력
    cv2.imshow('Multi-Person Squat Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
