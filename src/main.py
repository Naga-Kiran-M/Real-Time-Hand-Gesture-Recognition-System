import cv2
import mediapipe as mp
import tensorflow as tf
from utils.gesture_recognition import recognize_gesture  # Import gesture recognition function

# Load the trained model (ensure to load it once at the start)
model = tf.keras.models.load_model("path_to_your_model.h5")  # Replace with your model path

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Set to 2 to detect up to two hands
    min_detection_confidence=0.9,  # Increased detection confidence for better accuracy
    min_tracking_confidence=0.9    # Increased tracking confidence for smoother hand tracking
)
mp_draw = mp.solutions.drawing_utils  # Used for drawing hand landmarks

# Start webcam feed
cap = cv2.VideoCapture(0)  # Access the first camera device

# Set video resolution to reduce lag and improve performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height to 480

# Create a resizable window
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)  # Allow the window to be resized

while True:
    try:
        # Capture frame from the webcam
        success, frame = cap.read()
        if not success:
            break  # If frame is not read successfully, break the loop

        # Flip the image horizontally for a mirror view and convert BGR to RGB
        frame = cv2.flip(frame, 1)  # Flip the frame to get mirror effect (real-time interaction)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for MediaPipe

        # Process the frame to detect hands and track landmarks
        result = hands.process(rgb_frame)

        # If hand landmarks are detected, draw them on the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Recognize gesture based on landmarks
                gesture = recognize_gesture(hand_landmarks.landmark, model)  # Pass the model for prediction
                
                # Display the recognized gesture on the frame at the top-left corner
                cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize the frame if necessary (to adjust the output window size)
        resized_frame = cv2.resize(frame, (640, 480))  # Resize the frame to match the desired resolution

        # Display the processed frame with landmarks in the resizable window
        cv2.imshow("Hand Tracking", resized_frame)

        # Wait for key press and exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF  # Wait for a key event (1ms)
        if key == ord('q'):  # If 'q' key is pressed, break the loop
            break

    except KeyboardInterrupt:  # Handle manual interruption gracefully (Ctrl + C)
        print("Program interrupted.")
        break  # Exit the loop if keyboard interrupt occurs

# Release the webcam and close all OpenCV windows
cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
