import numpy as np
import pandas as pd

from models.model_loader import ModelLoader
from utils.preprocessing import preprocess
from core.config import UNCERTAIN_LOW, UNCERTAIN_HIGH
from core.logger import logger


class PredictionService:

    def __init__(self):

        # Load all saved files
        loader = ModelLoader()

        self.model = loader.model
        self.scaler = loader.scaler
        self.features = loader.features
        self.ood_model = loader.ood_model

    # Main prediction function
    def predict(self, df: pd.DataFrame):

        try:

            logger.info("Starting prediction pipeline")

            # Preprocess data
            df = preprocess(df, self.features)

            # Scale the data
            X = self.scaler.transform(df)

            # Get prediction probability

            # Some models support probability prediction
            if hasattr(self.model, "predict_proba"):

                # Get probability of Attack class
                probs = self.model.predict_proba(X)[:, 1]

            else:

                # If probability is not supported
                # use normal predictions instead

                preds = self.model.predict(X)

                probs = []

                for p in preds:

                    if p == 1:
                        probs.append(1.0)
                    else:
                        probs.append(0.0)

                probs = np.array(probs).astype(float)

            # OOD Detection

            # OOD = Out Of Distribution
            # Means data looks different from training data

            if self.ood_model is not None:

                # Isolation Forest gives:
                # -1 = anomaly
                #  1 = normal

                predictions = self.ood_model.predict(X)

                ood_flags = []

                for value in predictions:

                    if value == -1:
                        ood_flags.append(True)
                    else:
                        ood_flags.append(False)

            else:

                # If no OOD model exists
                ood_flags = [False] * len(probs)

            # Store final results
            results = []

            # Process each row

            for i in range(len(probs)):

                attack_prob = float(probs[i])

                is_ood = bool(ood_flags[i])

                # Convert probability into percentage
                risk_score = attack_prob * 100


                # Decide Status


                if attack_prob < UNCERTAIN_LOW:

                    status = "Normal"

                    confidence = (1.0 - attack_prob) * 100

                elif attack_prob > UNCERTAIN_HIGH:

                    status = "Attack"

                    confidence = attack_prob * 100

                else:

                    status = "Uncertain"

                    # Lower confidence near 0.5
                    confidence = (
                        1.0 - 2.0 * abs(attack_prob - 0.5)
                    ) * 100


                # OOD Override


                # If data is unusual and model says Normal
                # then mark it as Uncertain

                if is_ood and status == "Normal":

                    status = "Uncertain"

                    confidence = min(confidence, 50.0)


                # Risk Level


                if risk_score < 30:

                    risk_level = "LOW"

                elif risk_score < 70:

                    risk_level = "MEDIUM"

                else:

                    risk_level = "HIGH"

                # If OOD + Uncertain
                # then force MEDIUM risk

                if is_ood and status == "Uncertain":

                    risk_level = "MEDIUM"


                # Save Result


                result = {
                    "id": i,
                    "status": status,
                    "confidence": round(confidence, 2),
                    "risk_score": round(risk_score, 2),
                    "risk_level": risk_level,
                    "is_ood": is_ood,
                }

                results.append(result)

            logger.info("Prediction completed successfully")

            return results

        except Exception as e:

            logger.error(f"Prediction error: {e}")

            raise Exception(f"Prediction failed: {e}")