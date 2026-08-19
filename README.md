# Finger-Position-Based-Object-Tracking
Detects the direction where a finger is pointing in webcam video and continuously tracks the movements of surrounding objects.

## How It Works
This project uses Google Mediapipe Hand Landmarker and OpenCV to detect finger pointing and smooth object tracking in real-time, with a webcam. When you point at an object and press the capture button, the frame is saved and processed by the AI model, which outlines each finger. The script takes the coordinates of the index tip and the index dip to create a direction vector, and a bounding box is placed at the end of the direction vector to capture the region of interest.

The object tracking does not use an AI model and instead uses the CRST algorithm, in OpenCV. More about CRST here: https://docs.opencv.org/4.1.1/d2/da2/classcv_1_1TrackerCSRT.html

The CRST tracking box continuously updates until the object is no longer seen in the frame.

## Dependencies
Google Mediapipe Hand Landmarker Model: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models
OpenCV and Python 3.9+

## How To Run
Place the Google Mediapipe Hand Landmarker Model (.task file) into the same folder as the Python script.
When you run the script, the webcam will automatically open, allowing you to view the video.
Press "Q" to close the webcam and script.
Point your index finger at an object in view of the camera, and press "C." If successful, a green bounding box should appear at the pointed object, and the bounding box should be tracking until the object is no longer in view of the camera. An image will also be saved into the folder as "hand_landmarks.jpg", and the image shows you your finger landmarks, direction vector of the pointer finger, and initial bounding box.

## Potential Updates/Improvements for the future:
Bounding box could reappear if an object that was lost reappears in the frame
Continuous tracking around a finger by putting the Hand Landmarker model in VIDEO mode, however this may drop framerate
Different kinds of object tracking based on different finger gestures
