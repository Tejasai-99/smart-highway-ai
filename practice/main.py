from ultralytics import YOLO
import cv2
import supervision as sv

print("Program started")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/video2.mp4")

# Check video
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

print("Video opened successfully")

# Initialize tracker
tracker = sv.ByteTrack()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    # YOLO detection
    results = model(frame)[0]

    # Convert detections
    detections = sv.Detections.from_ultralytics(results)

    # Tracking
    detections = tracker.update_with_detections(detections)

    # Draw tracking boxes
    for i in range(len(detections.xyxy)):

        x1, y1, x2, y2 = detections.xyxy[i].astype(int)

        tracker_id = detections.tracker_id[i]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            frame,
            f"ID {tracker_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    # Show video
    cv2.imshow("Tracking System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()