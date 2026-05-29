from ultralytics import YOLO
import cv2
import supervision as sv
from collections import defaultdict

# ---------------- MODEL ---------------- #

model = YOLO("yolov8n.pt")

# ---------------- VIDEOS ---------------- #

capA = cv2.VideoCapture("videos/video2.mp4")
capB = cv2.VideoCapture("videos/video2.mp4")

# ---------------- TRACKERS ---------------- #

trackerA = sv.ByteTrack()
trackerB = sv.ByteTrack()

# ---------------- STORAGE ---------------- #

vehicle_counts_A = defaultdict(int)

vehicle_labels_A = {}

stored_vehicles = set()

# ---------------- EXIT LINE ---------------- #

EXIT_LINE_Y = 250

# ---------------- COLOR DETECTION ---------------- #

def detect_color(crop):

    crop = cv2.resize(crop, (50, 50))

    avg_color = crop.mean(axis=0).mean(axis=0)

    blue, green, red = avg_color

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

    retA, frameA = capA.read()
    retB, frameB = capB.read()

    if not retA or not retB:
        break

    # ---------------- CAMERA A ---------------- #

    resultsA = model(frameA, verbose=False)[0]

    detectionsA = sv.Detections.from_ultralytics(resultsA)

    detectionsA = trackerA.update_with_detections(detectionsA)

    # Draw exit line
    cv2.line(
        frameA,
        (0, EXIT_LINE_Y),
        (frameA.shape[1], EXIT_LINE_Y),
        (0, 0, 255),
        3
    )

    for i in range(len(detectionsA.xyxy)):

        x1, y1, x2, y2 = detectionsA.xyxy[i].astype(int)

        tracker_id = detectionsA.tracker_id[i]

        class_id = detectionsA.class_id[i]

        class_name = model.names[class_id]

        if class_name not in ["car", "truck", "bus", "motorbike"]:
            continue

        crop = frameA[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        color = detect_color(crop)

        # Short names
        if class_name == "car":
            short_name = "c"

        elif class_name == "truck":
            short_name = "t"

        elif class_name == "bus":
            short_name = "b"

        else:
            short_name = "m"

        # Create label
        if tracker_id not in vehicle_labels_A:

            vehicle_counts_A[f"{color}{short_name}"] += 1

            count = vehicle_counts_A[f"{color}{short_name}"]

            label = f"{color}{short_name}{count}"

            vehicle_labels_A[tracker_id] = label

        else:

            label = vehicle_labels_A[tracker_id]

        # Draw box
        cv2.rectangle(frameA, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            frameA,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        # Exit line crossing
        center_y = (y1 + y2) // 2

        if center_y > EXIT_LINE_Y:

            stored_vehicles.add(label)

    # ---------------- CAMERA B ---------------- #

    resultsB = model(frameB, verbose=False)[0]

    detectionsB = sv.Detections.from_ultralytics(resultsB)

    detectionsB = trackerB.update_with_detections(detectionsB)

    for i in range(len(detectionsB.xyxy)):

        x1, y1, x2, y2 = detectionsB.xyxy[i].astype(int)

        class_id = detectionsB.class_id[i]

        class_name = model.names[class_id]

        if class_name not in ["car", "truck", "bus", "motorbike"]:
            continue

        crop = frameB[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        color = detect_color(crop)

        # Short names
        if class_name == "car":
            short_name = "c"

        elif class_name == "truck":
            short_name = "t"

        elif class_name == "bus":
            short_name = "b"

        else:
            short_name = "m"

        # Temporary label
        temp_label = f"{color}{short_name}1"

        # SAFE check
        if temp_label in stored_vehicles:

            cv2.putText(
                frameB,
                "SAFE",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

            print(f"SAFE: {temp_label}")

        # Draw box
        cv2.rectangle(frameB, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.putText(
            frameB,
            temp_label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )

    # ---------------- SHOW WINDOWS ---------------- #

    cv2.imshow("Camera A", frameA)
    cv2.imshow("Camera B", frameB)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- RELEASE ---------------- #

capA.release()
capB.release()

cv2.destroyAllWindows()