import cv2

# hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17)                  # palm
]

# hand landmark indices
INDEX_FINGER_TIP = 8
RING_FINGER_TIP = 16
INDEX_FINGER_MCP = 5
RING_FINGER_MCP = 13


def draw_hand_landmarks(frame, hand_landmarks):
    if not hand_landmarks:
        return

    height, width = frame.shape[:2]

    # draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # draw landmark points
    for lm in hand_landmarks:
        point = (int(lm.x * width), int(lm.y * height))
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def calculate_finger_ratio(frame, hand_landmarks):
    if not hand_landmarks:
        return None

    # get coordinates
    index_finger_tip = hand_landmarks[INDEX_FINGER_TIP]
    ring_finger_tip = hand_landmarks[RING_FINGER_TIP]
    index_finger_mcp = hand_landmarks[INDEX_FINGER_MCP]
    ring_finger_mcp = hand_landmarks[RING_FINGER_MCP]

    # draw landmarks on the image
    draw_hand_landmarks(frame, hand_landmarks)

    # calculate lengths
    index_length = ((index_finger_tip.x - index_finger_mcp.x) **
                    2 + (index_finger_tip.y - index_finger_mcp.y) ** 2) ** 0.5
    ring_length = ((ring_finger_tip.x - ring_finger_mcp.x) **
                   2 + (ring_finger_tip.y - ring_finger_mcp.y) ** 2) ** 0.5

    # ratio-sratio
    if ring_length != 0:
        return index_length / ring_length
    else:
        return 1.0
