from fastapi import FastAPI
from routers.predict import router

app = FastAPI(title="House Rent Prediction API", version="1.0.0")

# Include the prediction router
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "House Rent API is live! Go to /docs"}