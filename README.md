# How to run

1. install `hands_landmark.task` into the `models/` directory from [ai.google.dev](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) either directly from the website, or via curl:
    ```bash
    mkdir -p models
    curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    ```
2. create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. run the code:
    ```bash
    python3 main.py
    ```

enjoy