from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predictor import SentimentPredictor

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyse sentiment of product reviews",
    version="1.0"
)

# Load model once when server starts — not on every request
# This is critical for performance
predictor = SentimentPredictor()

# Request schema — Pydantic validates this automatically
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

# Response schema
class SentimentResponse(BaseModel):
    label:      str
    confidence: float
    scores:     dict

@app.get("/")
def root():
    return {"message": "Sentiment API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: ReviewRequest):
    try:
        result = predictor.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for text in request.texts:
        results.append(predictor.predict(text))
    return results