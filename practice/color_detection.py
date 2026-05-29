from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/video2.mp4")

# Function to detect dominant color
def detect_color(crop):

    # Resize crop
    crop = cv2.resize(crop, (50, 50))

    # Average color
    avg_color = crop.mean(axis=0).mean(axis=0)

    blue, green, red = avg_color

    # Simple color rules
    if red > green and red > blue:
        return "Red"

    elif blue > red and blue > green:
        return "Blue"

    elif green > red and green > blue:
        return "Green"

    elif red > 150 and green > 150 and blue > 150:
        return "White"

    elif red < 80 and green < 80 and blue < 80:
        return "Black"

    else:
        return "Unknown"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame)[0]

    for box in results.boxes:

        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class ID
        cls_id = int(box.cls[0])

        # Get class name
        class_name = model.names[cls_id]

        # Detect only vehicles
        if class_name in ["car", "truck", "bus", "motorbike"]:

            # Crop vehicle
            crop = frame[y1:y2, x1:x2]

            # Skip empty crop
            if crop.size == 0:
                continue

            # Detect color
            color = detect_color(crop)

            # Create label
            label = f"{color}_{class_name}"

            print(label)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Put label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    # Show video
    cv2.imshow("Color Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()