from ultralytics import YOLO
import cv2
import supervision as sv

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open videos
capA = cv2.VideoCapture("videos/video2.mp4")
capB = cv2.VideoCapture("videos/video2.mp4")

# Trackers
trackerA = sv.ByteTrack()
trackerB = sv.ByteTrack()

# Store IDs
cameraA_ids = set()
cameraB_ids = set()

while True:

    # Read frames
    retA, frameA = capA.read()
    retB, frameB = capB.read()

    if not retA or not retB:
        break

    # Detect Camera A
    resultA = model(frameA)[0]
    detectionsA = sv.Detections.from_ultralytics(resultA)
    detectionsA = trackerA.update_with_detections(detectionsA)

    # Detect Camera B
    resultB = model(frameB)[0]
    detectionsB = sv.Detections.from_ultralytics(resultB)
    detectionsB = trackerB.update_with_detections(detectionsB)

    # Store Camera A IDs
    for tracker_id in detectionsA.tracker_id:
        cameraA_ids.add(tracker_id)

    # Store Camera B IDs
    for tracker_id in detectionsB.tracker_id:
        cameraB_ids.add(tracker_id)

    # Draw boxes Camera A
    for i in range(len(detectionsA.xyxy)):

        x1, y1, x2, y2 = detectionsA.xyxy[i].astype(int)
        tracker_id = detectionsA.tracker_id[i]

        cv2.rectangle(frameA, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frameA,
            f"A ID {tracker_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # Draw boxes Camera B
    for i in range(len(detectionsB.xyxy)):

        x1, y1, x2, y2 = detectionsB.xyxy[i].astype(int)
        tracker_id = detectionsB.tracker_id[i]

        cv2.rectangle(frameB, (x1,y1), (x2,y2), (255,0,0), 2)

        cv2.putText(
            frameB,
            f"B ID {tracker_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,0,0),
            2
        )

    # Compare IDs
    common_ids = cameraA_ids.intersection(cameraB_ids)

    print("Vehicles moved from A to B:", common_ids)

    # Show windows
    cv2.imshow("Camera A", frameA)
    cv2.imshow("Camera B", frameB)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capA.release()
capB.release()
cv2.destroyAllWindows()