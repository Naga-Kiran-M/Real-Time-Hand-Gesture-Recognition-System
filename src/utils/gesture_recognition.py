import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load the pre-trained TensorFlow Lite gesture recognition model
gesture_model = tf.lite.Interpreter(model_path="gesture_recognition_model.tflite")
gesture_model.allocate_tensors()  # Allocate tensors for the model

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Allow up to 2 hands to be tracked
    min_detection_confidence=0.9,  # Minimum confidence threshold for detection
    min_tracking_confidence=0.9  # Minimum confidence threshold for tracking
)
mp_draw = mp.solutions.drawing_utils  # Utility for drawing landmarks

def load_gesture_labels(label_file_path="gesture_labels.txt"):
    """
    Loads the gesture labels from a file into a list.
    :param label_file_path: Path to the gesture labels file.
    :return: List of gesture labels.
    """
    with open(label_file_path, "r") as f:
        labels = f.readlines()
    return [label.strip() for label in labels]

# Load gesture labels
gesture_labels = load_gesture_labels()

def predict_gesture(hand_landmarks):
    """
    Predicts the gesture based on the hand landmarks by processing the input for the model.
    
    :param hand_landmarks: List of hand landmarks for a detected hand.
    :return: Gesture label predicted by the model.
    """
    # Flatten the hand landmarks (x, y, z coordinates) into a list
    landmarks = []
    for landmark in hand_landmarks:
        landmarks.extend([landmark.x, landmark.y, landmark.z])  # Flattening each landmark's (x, y, z)

    # Convert landmarks to a numpy array and preprocess
    input_data = np.array(landmarks, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension (required by the model)

    # Set the input tensor for the model
    input_details = gesture_model.get_input_details()
    gesture_model.set_tensor(input_details[0]['index'], input_data)

    # Run inference with the model
    gesture_model.invoke()

    # Get the output tensor (gesture class probabilities)
    output_details = gesture_model.get_output_details()
    gesture_probabilities = gesture_model.get_tensor(output_details[0]['index'])

    # Find the gesture class with the highest probability (most likely gesture)
    gesture_class = np.argmax(gesture_probabilities)
    
    # Return the gesture label based on the predicted class index
    return gesture_labels[gesture_class]  # Return the actual gesture label

# Start the webcam feed
cap = cv2.VideoCapture(0)  # Use default webcam

while True:
    success, frame = cap.read()  # Capture a frame from the webcam
    if not success:  # If the frame isn't read successfully, exit the loop
        break

    # Flip the image horizontally for a mirror effect (as if looking at a mirror)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB for MediaPipe processing
    
    # Process the frame for hand landmarks using MediaPipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:  # If landmarks are detected for any hand
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections between them
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Predict the gesture from the hand landmarks
            gesture_label = predict_gesture(hand_landmarks.landmark)
            print("Predicted Gesture:", gesture_label)  # Print the predicted gesture label

            # Display the gesture label on the frame
            cv2.putText(frame, f"Gesture: {gesture_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed frame with hand landmarks
    cv2.imshow("Hand Tracking", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
