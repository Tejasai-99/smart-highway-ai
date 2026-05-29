from ultralytics import YOLO
import cv2
import supervision as sv

# Load YOLO
model = YOLO("yolov8n.pt")

# Exit line position
EXIT_LINE_Y = 250

def run_detection(video_path):

    # Tracker inside function
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)

    # Store counted IDs
    counted_ids = set()

    total_vehicles = 0

    car_count = 0
    truck_count = 0
    bike_count = 0
    bus_count = 0

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Process every 10th frame
        if frame_count % 10 != 0:
            continue

        # Resize frame
        frame = cv2.resize(frame, (640, 384))

        # YOLO Detection
        results = model(frame, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)

        # Tracking
        detections = tracker.update_with_detections(
            detections
        )

        for i in range(len(detections.xyxy)):

            x1, y1, x2, y2 = detections.xyxy[i].astype(int)

            tracker_id = detections.tracker_id[i]

            class_id = detections.class_id[i]

            class_name = model.names[class_id]

            # Only vehicles
            if class_name not in [
                "car",
                "truck",
                "bus",
                "motorbike"
            ]:
                continue

            # Vehicle center
            center_y = (y1 + y2) // 2

            # ONLY count after crossing line
            if (
                center_y > EXIT_LINE_Y
                and tracker_id not in counted_ids
            ):

                counted_ids.add(tracker_id)

                total_vehicles += 1

                if class_name == "car":
                    car_count += 1

                elif class_name == "truck":
                    truck_count += 1

                elif class_name == "motorbike":
                    bike_count += 1

                elif class_name == "bus":
                    bus_count += 1

    cap.release()

    return {
        "total": total_vehicles,
        "cars": car_count,
        "trucks": truck_count,
        "bikes": bike_count,
        "buses": bus_count
    }