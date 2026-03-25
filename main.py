import vision.hand_tracker as hand_tracker
import mediapipe as mp
import cv2
import time
from pathlib import Path
from the_magic import the_calculation, the_more_forgiving_calculation


MODEL_PATH = Path(__file__).resolve().parent / \
    "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "no MediaPipe hand landmarker model at "
        f"{MODEL_PATH} :( \n Download a .task model and place it there."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

# process the video feed frame by frame
cap = cv2.VideoCapture(0)
start_time = time.monotonic()


with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("TRACE: no read from webcam :(")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.hand_landmarks:
            ratio = hand_tracker.calculate_finger_ratio(
                frame, result.hand_landmarks[0]
            )
            if ratio is not None:
                print(f"TRACE: Finger ratio calculated: {ratio}")

            # draw the results as text
            if ratio is not None:
                cv2.putText(frame, f"The Ratio: {ratio:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3)
                cv2.putText(frame, f"Calculated Magic Value: {the_calculation(ratio):.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (70, 70, 225), 3)
                cv2.putText(frame, f"Forgiving Magic Value: {the_more_forgiving_calculation(ratio):.1f}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (40, 40, 225), 3)
        # display the resulting frame (the ratio)
        cv2.imshow('Hand Tracking', frame)

        # esc
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
