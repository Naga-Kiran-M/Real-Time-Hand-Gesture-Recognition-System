# Real-Time Hand Gesture Recognition System

## Overview
This **Real-Time Hand Gesture Recognition System** uses **MediaPipe**, **OpenCV**, and **TensorFlow** to detect and classify hand gestures in real time. The system provides a user-friendly interface for intuitive interaction, ideal for **AR/VR applications** and **AI-driven control systems**.

## Features
- **Gesture Detection**: Real-time recognition of 8 predefined hand gestures.
- **Deep Learning**: Powered by a pre-trained **TensorFlow** model for accurate gesture classification.
- **User Interface**: Clean UI with real-time feedback and FPS tracking.
- **Cross-platform Compatibility**: Developed using Python and works on various operating systems.
- **Modular Design**: Easy to extend and integrate with other AR/VR systems.

## Predefined Gestures
The system recognizes the following gestures:
1. **Fist**
2. **Open Hand**
3. **Peace Sign**
4. **Thumbs Up**
5. **Dog**
6. **Thumbs Down**
7. **Rock On**
8. **Punch**

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Real-Time-Hand-Gesture-Recognition-System.git
   ```

2. Install required dependencies:
   ```bash
   cd Real-Time-Hand-Gesture-Recognition-System
   pip install -r requirements.txt
   ```

## Usage
Run the system with the following command:

```bash
python main.py
```

The webcam will open, and the system will start recognizing hand gestures in real-time.

Press 'q' to exit the application.

## System Requirements
- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow

## Technologies Used
- **Python**: Core programming language.
- **OpenCV**: Image processing and computer vision.
- **MediaPipe**: Hand landmark detection.
- **TensorFlow**: Gesture classification using a pre-trained neural network.

## Model Training
The system uses a pre-trained TensorFlow model trained on hand gesture datasets. You can train a new model by providing labeled gesture images and adjusting the training parameters.

## Example Output
- **Gesture Label**: Displays the name of the recognized gesture.
- **Confidence Level**: Shows the model's confidence in its prediction.
- **FPS (Frames Per Second)**: Tracks real-time processing performance.

## Future Scope
- **Extended Gesture Library**: Add more gestures for advanced interaction.
- **Multi-hand Recognition**: Enhance the system to detect multiple hands at once.
- **Real-time Integration**: Integrate with AR/VR applications for immersive experiences.

## Contributing
Contributions are welcome! If you have ideas to improve the system, feel free to fork this repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References
- MediaPipe: https://google.github.io/mediapipe/
- OpenCV: https://opencv.org/
- TensorFlow: https://www.tensorflow.org/

## Contact
For inquiries or issues, feel free to open an issue in this repository or reach out via email: nagakiranm2021@gmail.com

Enjoy exploring the Real-Time Hand Gesture Recognition System!
