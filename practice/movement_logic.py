from ultralytics import YOLO
import cv2
import supervision as sv

# Load model
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

    retA, frameA = capA.read()
    retB, frameB = capB.read()

    if not retA or not retB:
        break

    # Camera A detection
    resultA = model(frameA)[0]
    detectionsA = sv.Detections.from_ultralytics(resultA)
    detectionsA = trackerA.update_with_detections(detectionsA)

    # Camera B detection
    resultB = model(frameB)[0]
    detectionsB = sv.Detections.from_ultralytics(resultB)
    detectionsB = trackerB.update_with_detections(detectionsB)

    # Save Camera A IDs
    for tracker_id in detectionsA.tracker_id:
        cameraA_ids.add(tracker_id)

    # Save Camera B IDs
    for tracker_id in detectionsB.tracker_id:
        cameraB_ids.add(tracker_id)

    # Compare IDs
    common_ids = cameraA_ids.intersection(cameraB_ids)

    # Print movement
    print("Vehicles moved from A to B:", common_ids)

    # Show videos
    cv2.imshow("Camera A", frameA)
    cv2.imshow("Camera B", frameB)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capA.release()
capB.release()
cv2.destroyAllWindows()