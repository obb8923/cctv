import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv8 모델 불러오기
model = YOLO('yolov8n.pt')  # YOLOv8 pre-trained 모델

# DeepSORT 객체 추적기 초기화
tracker = DeepSort(max_age=30)

# CCTV 영상 파일 열기
cap = cv2.VideoCapture('2p_switch.mp4')  # 분석할 CCTV 영상 경로

while cap.isOpened():  # 비디오 캡처가 열려 있는 동안 반복합니다.
    ret, frame = cap.read()  # 프레임을 읽습니다.
    if not ret:  # 프레임을 읽지 못하면 루프를 종료합니다.
        break

    # YOLOv8로 사람 탐지
    results = model(frame)  # 현재 프레임에서 객체를 탐지합니다.
    detections = []  # 탐지된 객체 정보를 저장할 리스트 초기화

    for result in results:  # 탐지 결과를 반복합니다.
        boxes = result.boxes  # 탐지된 박스 정보를 가져옵니다.
        for box in boxes:  # 각 박스에 대해 반복합니다.
            # 좌표 추출 및 변환
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 박스의 좌표를 가져옵니다.
            score = float(box.conf[0].cpu().numpy())  # 신뢰도를 가져옵니다.
            class_id = int(box.cls[0].cpu().numpy())  # 클래스 ID를 가져옵니다.
            
            # 사람만 탐지 (class_id == 0)
            if class_id == 0:  # 클래스 ID가 0인 경우(사람)만 처리합니다.
                # 너비와 높이 계산
                width = (x2 - x1)  # 박스의 너비 계산
                height = (y2 - y1)  # 박스의 높이 계산
                # DeepSORT가 기대하는 형식으로 저장
                raw_detections = [([x1, y1, width, height], score, class_id)]
                detections.extend(raw_detections)
    # 디버깅: detections의 구조를 출력
    print("Detections:", detections)  # 탐지된 객체 정보를 출력합니다.
    print("--------------------------------")
    # DeepSORT로 객체 추적
    tracked_objects = []  # tracked_objects 변수를 초기화합니다.
    if detections:  # detections가 비어있지 않은 경우에만 업데이트
        # 객체 추적 업데이트
        tracked_objects = tracker.update_tracks(detections, frame=frame)

    # 각 추적된 객체에 대해 정보 표시
    for track in tracked_objects:  # 추적된 객체를 반복합니다.
        if not track.is_confirmed() or track.time_since_update > 1:  # 객체가 확인되지 않았거나 업데이트가 1프레임 이상인 경우
            continue  # 다음 객체로 넘어갑니다.
        bbox = track.to_tlbr()  # (top, left, bottom, right) 형식으로 바운딩 박스 좌표를 가져옵니다.
        track_id = track.track_id  # 추적 ID를 가져옵니다.
        # 바운딩 박스를 프레임에 그립니다.
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # 추적 ID를 프레임에 표시합니다.
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # 결과 영상 출력
    cv2.imshow("CCTV Analysis", frame)  # 현재 프레임을 화면에 표시합니다.
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프를 종료합니다.
        break

cap.release()  # 비디오 캡처 객체를 해제합니다.
cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.
