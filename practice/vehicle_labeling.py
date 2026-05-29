from ultralytics import YOLO
import cv2
import supervision as sv
from collections import defaultdict

# ---------------- LOAD MODEL ---------------- #

model = YOLO("yolov8n.pt")

# ---------------- OPEN VIDEO ---------------- #

cap = cv2.VideoCapture("videos/video2.mp4")

# ---------------- TRACKER ---------------- #

tracker = sv.ByteTrack()

# ---------------- STORAGE ---------------- #

vehicle_counts = defaultdict(int)

stored_ids = set()

vehicle_labels = {}

# ---------------- EXIT LINE ---------------- #

# Change this value if needed
EXIT_LINE_Y = 700

# ---------------- COLOR DETECTION FUNCTION ---------------- #

def detect_color(crop):

    # Resize crop
    crop = cv2.resize(crop, (50, 50))

    # Get average color
    avg_color = crop.mean(axis=0).mean(axis=0)

    blue, green, red = avg_color

    # Simple color detection
    if red > green and red > blue:
        return "red"

    elif blue > red and blue > green:
        return "blue"

    elif green > red and green > blue:
        return "green"

    elif red > 150 and green > 150 and blue > 150:
        return "white"

    elif red < 80 and green < 80 and blue < 80:
        return "black"

    else:
        return "unknown"

# ---------------- MAIN LOOP ---------------- #

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame, verbose=False)[0]

    # Convert detections
    detections = sv.Detections.from_ultralytics(results)

    # Tracking
    detections = tracker.update_with_detections(detections)

    # ---------------- DRAW EXIT LINE ---------------- #

    cv2.line(
        frame,
        (0, EXIT_LINE_Y),
        (frame.shape[1], EXIT_LINE_Y),
        (0, 0, 255),
        3
    )

    # Exit line label
    cv2.putText(
        frame,
        "EXIT LINE",
        (20, EXIT_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ---------------- PROCESS DETECTIONS ---------------- #

    for i in range(len(detections.xyxy)):

        # Bounding box
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)

        # Tracker ID
        tracker_id = detections.tracker_id[i]

        # Class ID
        class_id = detections.class_id[i]

        # Class name
        class_name = model.names[class_id]

        # Detect only vehicles
        if class_name not in ["car", "truck", "bus", "motorbike"]:
            continue

        # Crop vehicle
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Detect color
        color = detect_color(crop)

        # ---------------- SHORT NAMES ---------------- #

        if class_name == "car":
            short_name = "c"

        elif class_name == "truck":
            short_name = "t"

        elif class_name == "bus":
            short_name = "b"

        else:
            short_name = "m"

        # ---------------- CREATE UNIQUE LABEL ---------------- #

        if tracker_id not in vehicle_labels:

            vehicle_counts[f"{color}{short_name}"] += 1

            count = vehicle_counts[f"{color}{short_name}"]

            label = f"{color}{short_name}{count}"

            vehicle_labels[tracker_id] = label

        else:

            label = vehicle_labels[tracker_id]

        # ---------------- DRAW BOX ---------------- #

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ---------------- DRAW LABEL ---------------- #

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ---------------- VEHICLE CENTER ---------------- #

        center_y = (y1 + y2) // 2

        # Draw center point
        cv2.circle(
            frame,
            ((x1 + x2) // 2, center_y),
            5,
            (255, 0, 0),
            -1
        )

        # ---------------- STORE AFTER EXIT ---------------- #

        if center_y > EXIT_LINE_Y and tracker_id not in stored_ids:

            stored_ids.add(tracker_id)

            print(f"STORED VEHICLE: {label}")

    # ---------------- SHOW OUTPUT ---------------- #

    cv2.imshow("Smart Vehicle Monitoring", frame)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- RELEASE ---------------- #

cap.release()

cv2.destroyAllWindows()