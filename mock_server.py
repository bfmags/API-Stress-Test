from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random
import asyncio
import os

# Read environment variables with default values
MIN_LATENCY = float(os.environ.get("MIN_LATENCY", "0.01"))
MAX_LATENCY = float(os.environ.get("MAX_LATENCY", "0.5"))
ERROR_RATE = float(os.environ.get("ERROR_RATE", "0.1"))

app = FastAPI()

@app.get("/test")
async def test_endpoint():
    """
    Simulate variable latency and error rate based on environment variables.
    """
    # Random delay
    await asyncio.sleep(random.uniform(MIN_LATENCY, MAX_LATENCY))
    # Random error
    if random.random() < ERROR_RATE:
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
