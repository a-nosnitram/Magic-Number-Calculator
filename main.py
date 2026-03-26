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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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

        # add hand image onto the screen from assets
        hand_image_path = str(
            Path(__file__).resolve().parent / "assets" / "hand.png")
        # Read with alpha channel if present
        hand_image = cv2.imread(hand_image_path, cv2.IMREAD_UNCHANGED)
        hand_image = cv2.resize(hand_image, (1000, 1000))
        hand_image = cv2.flip(hand_image, 1)
        # crop bottom 20% of the image
        hand_image = hand_image[:800, :1000]

        y_offset = frame.shape[0] - 1050  # 50px higher
        x_offset = frame.shape[1] - 1050  # 50px more to the left
        y1, y2 = y_offset, y_offset + hand_image.shape[0]
        x1, x2 = x_offset, x_offset + hand_image.shape[1]

        # If hand_image has alpha channel, blend it with 0.5 opacity
        opacity = 0.6
        if hand_image.shape[2] == 4:
            alpha_hand = (hand_image[:, :, 3] / 255.0) * opacity
            alpha_bg = 1.0 - alpha_hand
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha_hand * hand_image[:, :, c] +
                    alpha_bg * frame[y1:y2, x1:x2, c]
                )
        else:
            # If no alpha, just blend with 0.5 opacity
            frame[y1:y2, x1:x2] = (
                opacity * hand_image[:, :, :3] +
                (1 - opacity) * frame[y1:y2, x1:x2]
            )

        # put test under the hand image
        cv2.putText(frame, "Open your hand and make sure it.", (1150, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 3)
        cv2.putText(frame, "is facing the camera head on, no tilting", (1100, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 3)

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
