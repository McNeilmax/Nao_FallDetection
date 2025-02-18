import numpy as np

# Head control gains
HEAD_GAINS = {
    "kx": 0.5,         # Proportional gain for x-axis alignment
    "ky": 0.3,         # Proportional gain for y-axis alignment
    "speed": 0.11,     # Angular velocity
    "threshold": 0.15  # Control head following threshold in radians
}

# Body alignment
BODY_ALIGNMENT = {
    "kp": 0.35,       # Proportional gain for body alignment
    "threshold": 0.2  # Threshold for alignment
}

# Walking parameters
WALKING = {
    "kp": 0.002,                # Proportional gain for approaching a person
    "goal_height": 105,         # Target distance from the detected person
    "walking_threshold": 10     # Threshold for stopping
}

# Video settings
VIDEO = {
    "resolution": 1,   # 320x240
    "color_space": 0,  # RGB
    "fps": 10          # Frames per second
}

# YOLO detection
DETECTION = {
    "model_path": "best.pt",    # Path to the YOLO model file
    "confidence_threshold": 0.7 # Min confidence for valid detection
}

# Head scanning positions
SCAN_POSITIONS = {
    "angles": np.arange(-1.9, 2, 0.1).tolist(),  # Head angles in radians
    "step_size": 0.1
}
