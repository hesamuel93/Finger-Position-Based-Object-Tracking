import cv2
import mediapipe as mp
import math

# Google Mediapipe Hand Landmarker Model
MODEL_PATH = "hand_landmarker.task"

# Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)

def detect_pointer(frame):
    h, w, _ = frame.shape
    bbox_width = int(w / 4)
    bbox_height = int(h / 4)
    bbox = None

    scale = 320 / w
    resized_w = 320
    resized_h = int(h * scale)

    lower_res = cv2.resize(frame, (resized_w, resized_h))
    rgb = cv2.cvtColor(lower_res, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Draw points
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )
            
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            for a, b in connections:
                x1 = int(hand_landmarks[a].x * w)
                y1 = int(hand_landmarks[a].y * h)
                x2 = int(hand_landmarks[b].x * w)
                y2 = int(hand_landmarks[b].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        # Creating vector for fingertip direction
        hand = result.hand_landmarks[0]
        tip = hand[8]
        dip = hand[7]

        hand_size = math.sqrt(
            (hand[9].x - hand[0].x) ** 2 +
            (hand[9].y - hand[0].y) ** 2
        )
        scale = hand_size * 4.0

        dx = tip.x - dip.x
        dy = tip.y - dip.y
        end_x = tip.x + dx * scale
        end_y = tip.y + dy * scale

        x1 = int(tip.x * w)
        y1 = int(tip.y * h)
        x2 = int(end_x * w)
        y2 = int(end_y * h)

        x1 = max(x1, 0)
        x1 = min(x1, w - 1)
        y1 = max(y1, 0)
        y1 = min(y1, h - 1)
        x2 = max(x2, 0)
        x2 = min(x2, w - 1)
        y2 = max(y2, 0)
        y2 = min(y2, h - 1)

        cv2.arrowedLine(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
            tipLength=0.15
        )

        #pointer_vector = ((x1, y1), (x2, y2))
        bx = int(x2 - bbox_width / 2)
        by = int(y2 - bbox_height / 2)
        bx = max(0, min(bx, w - bbox_width))
        by = max(0, min(by, h - bbox_height))

        bbox = (bx, by, bbox_width, bbox_height)

        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox_width, bbox[1] + bbox_height),
            (255, 0, 0),
            3
        )

    else:
        print("No hand detected.")

    cv2.imwrite('hand_landmarks.jpg', frame)
    return frame, bbox

def zoom(frame, bbox):
    if bbox is None:
        print("No object to zoom into.")
        return frame
    else:
        og_height, og_width, _ = frame.shape
        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            print("Bounding box has unstable dimensions")
            return frame
        cropped_frame = frame[y : y + h, x : x + w]
        frame = cv2.resize(cropped_frame, (og_width, og_height))
        return frame

def main():
    capture = cv2.VideoCapture(0)

    bbox = None
    tracker = None
    tracking_mode = False

    zoom_mode = False

    while capture.isOpened():
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

        if tracking_mode:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                # Draw tracking box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
                    
                # Tracking status
                cv2.putText(
                    frame,
                    "TRACKING",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:
                # Tracker lost object
                tracking_mode = False

            if zoom_mode:
                frame = zoom(frame, bbox)

        if key == ord('q'):
            tracking_mode = False
            zoom_mode = False
            break
        
        if key == ord('c'):
            frame, bbox = detect_pointer(frame)
            if bbox is not None:
                tracking_mode = True
                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)

        if key == ord('z'):
            zoom_mode = not zoom_mode

        cv2.imshow('Webcam Feed', frame)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()