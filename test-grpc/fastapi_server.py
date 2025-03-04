from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class SumRequest(BaseModel):
    number: int


class SumResponse(BaseModel):
    result: int


@app.post("/calculate_sum", response_model=SumResponse)
async def calculate_sum(request: SumRequest):
    if request.number < 0:
        raise HTTPException(
            status_code=400, detail="Number must be non-negative")

    # Calculate sum from 0 to n
    total = sum(range(request.number + 1))
    return SumResponse(result=total)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
