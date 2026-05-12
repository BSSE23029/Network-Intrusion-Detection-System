from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from services.prediction_service import PredictionService
from core.config import MAX_FILE_SIZE_MB
from core.logger import logger

router = APIRouter()

service = PredictionService()


# batch prediction API
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
        # converting into dataframe
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # prediction
        results = service.predict(df)

        # Count total results
        total = len(results)
        
        # Start counters from 0
        attack = 0
        uncertain = 0
        
        # Check each result one by one
        for r in results:
            # If status is Attack
            if r["status"] == "Attack":
                attack += 1
            # If status is Uncertain
            elif r["status"] == "Uncertain":
                uncertain += 1
        
        # Remaining are Normal
        normal = total - attack - uncertain
        
        # Avoid division by zero
        if total == 0:
            denom = 1
        else:
            denom = total
        
        # Print log message
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