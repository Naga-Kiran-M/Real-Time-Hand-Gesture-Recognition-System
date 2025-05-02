import cv2
import numpy as np
import time
import mediapipe as mp
import tensorflow as tf

# Load trained gesture recognition model and label encoder
model = tf.keras.models.load_model("models/hand_gesture_model_final.h5")
label_encoder = np.load("models/label_encoder_classes.npy", allow_pickle=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0

# UI Config
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 255, 0)
BOX_COLOR = (0, 0, 0)
LABEL_BOX_COLOR = (0, 0, 0)
TEXT_BACKGROUND_COLOR = (0, 0, 0)

# Function to bring window to the front
def bring_window_to_front(window_name):
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

def predict_gesture(frame):
    # Flip frame to create mirror effect
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Flatten 21 landmark points (x, y, z) into 63-length vector
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            input_data = np.array(landmarks).reshape(1, -1)  # Shape: (1, 63)

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

            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            gesture = GESTURE_LABELS.get(predicted_class, f"Unknown ({predicted_class})")
            confidence = np.max(prediction)

            # Draw landmarks
            mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Overlay gesture label with clean background
            cv2.rectangle(flipped_frame, (0, 0), (450, 60), LABEL_BOX_COLOR, -1)
            cv2.putText(flipped_frame, f"Gesture: {gesture} ({confidence:.2f})",
                        (10, 40), FONT, 1, TEXT_COLOR, 2)

            return flipped_frame, gesture, confidence

    # If no hand detected
    cv2.rectangle(flipped_frame, (0, 0), (270, 60), TEXT_BACKGROUND_COLOR, -1)
    cv2.putText(flipped_frame, "No hand detected", (10, 40), FONT, 1, (0, 0, 255), 2)
    return flipped_frame, "None", 0.0

# Main loop
cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Recognition", 1024, 768)  # Adjusted to 1024x768 for better balance
bring_window_to_front("Gesture Recognition")  # Bring the window to the front

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, gesture, confidence = predict_gesture(frame)

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Draw FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                FONT, 0.8, (255, 255, 0), 2)

    # Show the processed frame
    cv2.imshow("Gesture Recognition", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
