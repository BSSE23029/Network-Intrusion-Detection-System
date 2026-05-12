import os
import joblib
from core.config import (
    MODEL_PATH,
    CALIBRATED_MODEL_PATH,
    OOD_DETECTOR_PATH,
    SCALER_PATH,
    FEATURES_PATH,
)
from core.logger import logger


class ModelLoader:

    def __init__(self):

        try:
            logger.info("Loading ML artifacts...")

            # Prefer the calibrated model when present so probabilities feed
            # the Uncertain abstention band correctly. Fall back to the raw
            # best_model.pkl so the backend still boots before calibration
            # has been run.
            if os.path.exists(CALIBRATED_MODEL_PATH):
                logger.info("Using calibrated model")
                self.model = joblib.load(CALIBRATED_MODEL_PATH)
                self.is_calibrated = True
            else:
                logger.info(
                    "Calibrated model not found, falling back to best_model.pkl. "
                    "Run ML/scripts/calibrate.py to enable the Uncertain band."
                )
                self.model = joblib.load(MODEL_PATH)
                self.is_calibrated = False

            self.scaler = joblib.load(SCALER_PATH)
            self.features = joblib.load(FEATURES_PATH)

            # Optional OOD detector. When present, prediction_service uses it
            # to override confident-Normal verdicts on rows that don't look
            # like anything in the training distribution.
            if os.path.exists(OOD_DETECTOR_PATH):
                logger.info("Loading OOD detector")
                self.ood_model = joblib.load(OOD_DETECTOR_PATH)
            else:
                logger.info(
                    "OOD detector not found. "
                    "Run ML/scripts/train_ood.py to enable novelty checks."
                )
                self.ood_model = None

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise e
