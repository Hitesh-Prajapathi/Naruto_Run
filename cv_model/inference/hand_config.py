"""Calibration values for the production hand-sign recognizer.

Thresholds were selected from the existing validation split and checked on the
held-out test split.  Classes with clean, high-confidence separation are kept
strict; lower-confidence classes retain more recall but require longer temporal
confirmation.
"""

HAND_CONFIDENCE_THRESHOLDS = {
    "bird": 0.70,
    "boar": 0.80,
    # Dog needs a lower floor when its runner-up is one of the camera-tested
    # Dog confusions.  A paired guard in HandSignRecognizer keeps the common
    # Tiger -> Dog confusion on the original 0.80 threshold.
    "dog": 0.68,
    "dragon": 0.50,
    "hare": 0.32,
    "horse": 0.88,
    "monkey": 0.68,
    "ox": 0.70,
    "ram": 0.92,
    "rat": 0.45,
    "snake": 0.65,
    "tiger": 0.68,
    "zero": 0.45,
}

HAND_MARGIN_THRESHOLDS = {
    "bird": 0.25,
    "boar": 0.35,
    "dog": 0.40,
    "dragon": 0.20,
    "hare": 0.02,
    "horse": 0.50,
    "monkey": 0.25,
    "ox": 0.30,
    "ram": 0.55,
    "rat": 0.05,
    "snake": 0.20,
    "tiger": 0.25,
    "zero": 0.05,
}

# Accepted observations required in a five-frame evidence window. Neutral
# calibration can increase individual false-positive classes from 3 to 4.
DEFAULT_HAND_REQUIRED_VOTES = 3
