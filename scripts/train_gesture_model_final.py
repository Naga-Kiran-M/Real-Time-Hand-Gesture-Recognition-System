import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# # Use Keras from the standalone package if tensorflow.keras.callbacks is not resolving
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

# Load gesture data
# Load the gesture data and corresponding labels from the 'assets/gesture_data' folder
data_path = "assets/gesture_data"
X = np.load(os.path.join(data_path, "gesture_data.npy"))  # Features: (samples, 63)
y = np.load(os.path.join(data_path, "gesture_labels.npy"))  # Labels: (samples,)

# Encode labels
# Convert string labels into numeric form for model training
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Normalize input features
# Normalize the features to a range of [0, 1] by dividing by the max value in the dataset
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Build the model
# Create a simple feed-forward neural network with Dropout layers for regularization
model = keras.Sequential([
    keras.layers.Input(shape=(63,)),  # Input layer (features with shape 63)
    keras.layers.Dense(128, activation='relu'),  # First hidden layer with ReLU activation
    keras.layers.Dropout(0.3),  # Dropout for regularization (30% of the neurons are dropped)
    keras.layers.Dense(64, activation='relu'),  # Second hidden layer with ReLU activation
    keras.layers.Dropout(0.3),  # Another dropout layer
    keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
# The model uses 'Adam' optimizer, 'sparse_categorical_crossentropy' loss for multi-class classification,
# and 'accuracy' as the evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
os.makedirs("models", exist_ok=True)  # Ensure the models folder exists

# Save the model with a timestamp to maintain version control
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"models/hand_gesture_model_{timestamp}.h5"

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  # Stop training if validation loss doesn't improve for 10 epochs
    ModelCheckpoint(model_filename, save_best_only=True)  # Save only the best model during training
]

# Train the model
history = model.fit(
    X_train, y_train,  # Train on the training data
    validation_data=(X_test, y_test),  # Validate on the test data
    epochs=300,  # Number of epochs to train the model
    batch_size=64,  # Batch size for training
    callbacks=callbacks,  # Use the callbacks defined earlier
    verbose=1  # Display training progress
)

# Save label encoder (overwriting the previous one)
# Save the label encoder classes (the mapping from numeric labels to gesture names)
np.save("models/label_encoder_classes.npy", label_encoder.classes_)

# Check if 'hand_gesture_model_final.h5' exists and remove it if it does
if os.path.exists("models/hand_gesture_model_final.h5"):
    os.remove("models/hand_gesture_model_final.h5")

# Rename the saved model to 'hand_gesture_model_final.h5'
os.rename(model_filename, "models/hand_gesture_model_final.h5")

# Plot the training history
# Visualize training and validation accuracy and loss over epochs
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='green')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Adjust layout to avoid overlap and save the plot
plt.tight_layout()
plt.savefig("models/training_history.png")  # Save plot as a PNG file
plt.show()

# Print final message indicating successful training and model saving
print("✅ Model training complete. The latest model has been saved as `models/hand_gesture_model_final.h5`.")
