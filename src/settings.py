# main.py

import cv2
import mediapipe as mp
from config.settings import HAND_TRACKING, CAMERA_ID, WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT, DEBUG_MODE

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=HAND_TRACKING["mode"],
    max_num_hands=HAND_TRACKING["max_hands"],
    min_detection_confidence=HAND_TRACKING["detection_confidence"],
    min_tracking_confidence=HAND_TRACKING["tracking_confidence"]
)

mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Draw landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow(WINDOW_NAME, cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)))

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
