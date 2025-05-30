from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random
import asyncio

app = FastAPI()

@app.get("/test")
async def test_endpoint():
    """
    Simulate variable latency (10–500ms) and a 10% error rate.
    """
    # Random delay
    await asyncio.sleep(random.uniform(0.01, 0.5))
    # Random error
    if random.random() < 0.1:
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
