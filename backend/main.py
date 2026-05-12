from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.logger import logger

app = FastAPI(
    title="ML Cybersecurity API",
    version="1.0.0"
)

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://3.84.61.149:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routes
app.include_router(router)


@app.get("/")
def home():

    logger.info("Health check endpoint called")

    return {
        "message": "Backend is running successfully"
    }