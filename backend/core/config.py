import os

# backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
CALIBRATED_MODEL_PATH = os.path.join(MODELS_DIR, "calibrated_model.pkl")
OOD_DETECTOR_PATH = os.path.join(MODELS_DIR, "ood_detector.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")

# -------------------------
# FILE LIMIT CONFIG
# -------------------------
MAX_FILE_SIZE_MB = 50   # you can change this

# -------------------------
# UNCERTAINTY BAND
# Rows whose calibrated P(attack) falls inside [UNCERTAIN_LOW, UNCERTAIN_HIGH]
# are labeled "Uncertain" instead of being forced into Normal/Attack.
# -------------------------
UNCERTAIN_LOW = 0.40
UNCERTAIN_HIGH = 0.60
