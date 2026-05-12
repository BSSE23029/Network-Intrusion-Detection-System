from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from services.prediction_service import PredictionService
from core.config import MAX_FILE_SIZE_MB
from core.logger import logger

router = APIRouter()

service = PredictionService()


# batch prediction API-
@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        logger.info("File received for prediction")

        # File size check
        contents = await file.read()

        file_size_mb = len(contents) / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max allowed: {MAX_FILE_SIZE_MB}MB"
            )

        # convert to dataframe
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # prediction
        results = service.predict(df)

        # summary
        total = len(results)
        attack = sum(1 for r in results if r["status"] == "Attack")
        uncertain = sum(1 for r in results if r["status"] == "Uncertain")
        normal = total - attack - uncertain

        denom = total if total else 1

        logger.info("Returning response to frontend")

        return {
            "summary": {
                "total": total,
                "attack": attack,
                "normal": normal,
                "uncertain": uncertain,
                "attack_percentage": round((attack / denom) * 100, 2),
                "uncertain_percentage": round((uncertain / denom) * 100, 2)
            },
            "predictions": results
        }

    except Exception as e:

        logger.error(f"API error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )