import numpy as np
import pandas as pd

from models.model_loader import ModelLoader
from utils.preprocessing import preprocess
from core.config import UNCERTAIN_LOW, UNCERTAIN_HIGH
from core.logger import logger


class PredictionService:

    def __init__(self):

        loader = ModelLoader()

        self.model = loader.model
        self.scaler = loader.scaler
        self.features = loader.features
        self.ood_model = loader.ood_model

    # prediction logic
    def predict(self, df: pd.DataFrame):

        try:

            logger.info("Starting prediction pipeline")

            df = preprocess(df, self.features)

            X = self.scaler.transform(df)

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[:, 1]
            else:
                # Fallback: no probability support -> use hard predictions as
                # pseudo-probabilities (0.0 or 1.0). The Uncertain band will
                # never fire in this mode.
                preds = self.model.predict(X)
                probs = np.where(preds == 1, 1.0, 0.0).astype(float)

            # Run the OOD detector once per batch if available.
            # IsolationForest.predict returns -1 for anomalies, +1 otherwise.
            if self.ood_model is not None:
                ood_flags = self.ood_model.predict(X) == -1
            else:
                ood_flags = np.zeros(len(probs), dtype=bool)

            results = []

            for i in range(len(probs)):

                attack_prob = float(probs[i])
                is_ood = bool(ood_flags[i])

                risk_score = attack_prob * 100

                # Status comes from the calibrated probability, not from the
                # raw model.predict() output, so the Uncertain band is honored.
                if attack_prob < UNCERTAIN_LOW:
                    status = "Normal"
                    confidence = (1.0 - attack_prob) * 100
                elif attack_prob > UNCERTAIN_HIGH:
                    status = "Attack"
                    confidence = attack_prob * 100
                else:
                    status = "Uncertain"
                    # closer to 0.5 -> lower confidence; reaches 0 at exactly 0.5
                    confidence = (1.0 - 2.0 * abs(attack_prob - 0.5)) * 100

                # OOD override: a row that doesn't statistically resemble
                # anything in the training set should never be quietly labeled
                # Normal. We do NOT override Attack predictions so genuine
                # attack detections survive even when novelty is flagged.
                if is_ood and status == "Normal":
                    status = "Uncertain"
                    confidence = min(confidence, 50.0)

                if risk_score < 30:
                    risk_level = "LOW"
                elif risk_score < 70:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "HIGH"

                # Force MEDIUM risk on OOD-uncertain rows so the table reflects
                # genuine ambiguity rather than a misleading LOW bucket.
                if is_ood and status == "Uncertain":
                    risk_level = "MEDIUM"

                results.append({
                    "id": i,
                    "status": status,
                    "confidence": round(confidence, 2),
                    "risk_score": round(risk_score, 2),
                    "risk_level": risk_level,
                    "is_ood": is_ood,
                })

            logger.info("Prediction completed successfully")

            return results

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise Exception(f"Prediction failed: {e}")
