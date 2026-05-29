from ultralytics import YOLO
import cv2
import supervision as sv
import time

# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO("yolov8n.pt")

# =========================================================
# LOAD VIDEOS
# =========================================================

capA = cv2.VideoCapture("videos/video2.mp4")
capB = cv2.VideoCapture("videos/video2.mp4")

# =========================================================
# TRACKERS
# =========================================================

trackerA = sv.ByteTrack()
trackerB = sv.ByteTrack()

# =========================================================
# STORAGE
# =========================================================

# Stores:
# {
#   "blue": timestamp
# }

vehicle_database = {}

# Vehicles already matched
matched_vehicles = set()

# Avoid repeated SAFE prints
safe_printed = set()

# Vehicles already stored
stored_colors = set()

# =========================================================
# SETTINGS
# =========================================================

EXIT_LINE_Y = 250

TIMEOUT = 10

# =========================================================
# COLOR DETECTION
# =========================================================

def detect_color(crop):

    crop = cv2.resize(crop, (50, 50))

    avg_color = crop.mean(axis=0).mean(axis=0)

    blue, green, red = avg_color

    # Simple color logic
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

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    retA, frameA = capA.read()
    retB, frameB = capB.read()

    if not retA or not retB:
        break

    # =====================================================
    # CAMERA A
    # =====================================================

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

    # Exit line text
    cv2.putText(
        frameA,
        "EXIT LINE",
        (20, EXIT_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    for i in range(len(detectionsA.xyxy)):

        # Bounding box
        x1, y1, x2, y2 = detectionsA.xyxy[i].astype(int)

        # Class ID
        class_id = detectionsA.class_id[i]

        # Class name
        class_name = model.names[class_id]

        # Detect only vehicles
        if class_name not in ["car", "truck", "bus", "motorbike"]:
            continue

        # Crop vehicle
        crop = frameA[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Detect color
        color = detect_color(crop)

        # Draw bounding box
        cv2.rectangle(
            frameA,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        # Draw label
        cv2.putText(
            frameA,
            color,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        # Vehicle center
        center_y = (y1 + y2) // 2

        # Draw center point
        cv2.circle(
            frameA,
            ((x1 + x2)//2, center_y),
            5,
            (255,0,0),
            -1
        )

        # =================================================
        # STORE VEHICLE AFTER CROSSING LINE
        # =================================================

        if center_y > EXIT_LINE_Y and color not in stored_colors:

            stored_colors.add(color)

            vehicle_database[color] = time.time()

            print(f"STORED: {color}")

    # =====================================================
    # CAMERA B
    # =====================================================

    resultsB = model(frameB, verbose=False)[0]

    detectionsB = sv.Detections.from_ultralytics(resultsB)

    detectionsB = trackerB.update_with_detections(detectionsB)

    for i in range(len(detectionsB.xyxy)):

        x1, y1, x2, y2 = detectionsB.xyxy[i].astype(int)

        class_id = detectionsB.class_id[i]

        class_name = model.names[class_id]

        # Only vehicles
        if class_name not in ["car", "truck", "bus", "motorbike"]:
            continue

        # Crop vehicle
        crop = frameB[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Detect color
        color = detect_color(crop)

        # =================================================
        # SAFE CHECK
        # =================================================

        if color in vehicle_database:

            matched_vehicles.add(color)

            # Print SAFE only once
            if color not in safe_printed:

                print(f"SAFE: {color}")

                safe_printed.add(color)

            # Draw SAFE text
            cv2.putText(
                frameB,
                "SAFE",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

        # Draw box
        cv2.rectangle(
            frameB,
            (x1, y1),
            (x2, y2),
            (255,0,0),
            2
        )

        # Draw label
        cv2.putText(
            frameB,
            color,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )

    # =====================================================
    # ALERT CHECK
    # =====================================================

    current_time = time.time()

    for vehicle, exit_time in vehicle_database.items():

        if vehicle not in matched_vehicles:

            elapsed = current_time - exit_time

            if elapsed > TIMEOUT:

                print(f"ALERT: {vehicle} missing!")

                matched_vehicles.add(vehicle)

    # =====================================================
    # SHOW WINDOWS
    # =====================================================

    cv2.imshow("Camera A", frameA)

    cv2.imshow("Camera B", frameB)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# RELEASE
# =========================================================

capA.release()
capB.release()

cv2.destroyAllWindows()