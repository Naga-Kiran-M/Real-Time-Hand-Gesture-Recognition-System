# ---------------------------------------------------
# AR/VR Gesture Data Collector - Version 4 (Final)
# ---------------------------------------------------
# Purpose:
# - Collect hand gesture landmark data with MediaPipe
# - Overwrites previous gesture dataset for consistency
# - Saves to fixed .npy filenames for model training
# ---------------------------------------------------

import sys
import subprocess
import pkg_resources
import os
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime  # For optional timestamps if needed

# --- Auto-install required packages ---
required = {"opencv-python", "mediapipe", "numpy"}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    python = sys.executable
    subprocess.check_call([python, "-m", "pip", "install", *missing])

# --- Define gesture labels (can be customized) ---
GESTURE_LABELS = {
    0: "Fist",
    1: "Open Hand",
    2: "Peace Sign",
    3: "Thumbs Up",
    4: "Dog",
    5: "Thumbs Down",
    6: "Rock ON!!!",
    7: "Punch"
}

# --- Number of samples to collect per gesture ---
samples_per_gesture = 100

# --- Folder to save collected data ---
output_dir = "assets/gesture_data"
os.makedirs(output_dir, exist_ok=True)

# --- File names for fixed overwrite behavior ---
data_file = os.path.join(output_dir, "gesture_data.npy")
label_file = os.path.join(output_dir, "gesture_labels.npy")

# --- Setup MediaPipe Hands and Drawing Utilities ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access the webcam. Please check your camera index.")

# --- Make display window topmost ---
cv2.namedWindow("Gesture Capture", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gesture Capture", cv2.WND_PROP_TOPMOST, 1)

# --- Display gesture menu ---
print("Available Gestures:")
for k, v in GESTURE_LABELS.items():
    print(f"{k} - {v}")

# --- Initialize data containers ---
all_data = []
all_labels = []

try:
    for gesture_label in GESTURE_LABELS:
        print(f"\n🔄 Starting: {GESTURE_LABELS[gesture_label]}")
        input("👉 Press Enter when ready...")

        collected = 0
        attempts = 0
        max_attempts = 500  # Prevent infinite loop

        while collected < samples_per_gesture and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame grab failed.")
                continue

            # Flip for mirror effect and convert to RGB
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            attempts += 1

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Flatten (x, y, z) coords into 1D array
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    all_data.append(landmarks)
                    all_labels.append(gesture_label)
                    collected += 1

                    # Draw hand and info
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f"Samples: {collected}/{samples_per_gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Prompt when hand not detected
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Gesture Capture", frame)

            # Exit early on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Exiting capture...")
                break

        if collected == 0:
            print(f"❌ No samples collected for: {GESTURE_LABELS[gesture_label]}")
        else:
            print(f"✅ Completed: {collected} samples for {GESTURE_LABELS[gesture_label]}")

finally:
    # --- Cleanup resources ---
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# --- Save collected data to fixed .npy files ---
if all_data and all_labels:
    np.save(data_file, np.array(all_data))
    np.save(label_file, np.array(all_labels))
    print(f"\n📁 Data saved to:\n  - {data_file}\n  - {label_file}")
else:
    print("⚠️ No gesture data collected. Nothing saved.")
