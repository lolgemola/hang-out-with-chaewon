from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from typing import Optional

# Tunable thresholds
HAPPY_CORNER_LIFT = 0.005
SURPRISED_BROW_HEIGHT = 0.35
SURPRISED_MOUTH_GAP = 0.03
ANGRY_BROW_DISTANCE = 0.08
ANGRY_BROW_DROP = 0.35


def _finger_extended(lm, tip: int, base: int) -> bool:
    return lm[tip].y < lm[base].y


def _classify_hand(lm) -> str:
    index = _finger_extended(lm, 8, 5)
    middle = _finger_extended(lm, 12, 9)
    ring = _finger_extended(lm, 16, 13)
    pinky = _finger_extended(lm, 20, 17)
    thumb_up = lm[4].y < lm[0].y

    if index and middle and ring and pinky:
        return "open_palm"
    if not index and not middle and not ring and not pinky:
        if thumb_up:
            return "thumbs_up"
        return "fist"
    if index and middle and not ring and not pinky:
        return "peace_sign"
    return ""


def classify(
    face_landmarks: Optional[NormalizedLandmarkList],
    hand_landmarks: Optional[NormalizedLandmarkList],
) -> str:
    if hand_landmarks is not None:
        hand_label = _classify_hand(hand_landmarks.landmark)
        if hand_label:
            return hand_label

    if face_landmarks is None:
        return "neutral"

    lm = face_landmarks.landmark

    mouth_centre = lm[13].y
    left_corner = lm[61].y
    right_corner = lm[291].y
    upper_lip = lm[13].y
    lower_lip = lm[14].y
    left_brow_outer = lm[70].y
    right_brow_outer = lm[300].y
    left_brow_inner = lm[55]
    right_brow_inner = lm[285]

    # happy: mouth corners higher (lower y) than mouth centre
    if left_corner < mouth_centre - HAPPY_CORNER_LIFT and right_corner < mouth_centre - HAPPY_CORNER_LIFT:
        return "happy"

    # surprised: brows raised high AND mouth open wide
    if (left_brow_outer < SURPRISED_BROW_HEIGHT and right_brow_outer < SURPRISED_BROW_HEIGHT
            and (lower_lip - upper_lip) > SURPRISED_MOUTH_GAP):
        return "surprised"

    # angry: inner brows pulled down and close together
    brow_distance = abs(left_brow_inner.x - right_brow_inner.x)
    if (brow_distance < ANGRY_BROW_DISTANCE
            and left_brow_inner.y > ANGRY_BROW_DROP
            and right_brow_inner.y > ANGRY_BROW_DROP):
        return "angry"

    return "neutral"
