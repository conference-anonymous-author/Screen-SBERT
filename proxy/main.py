"""FastAPI application entry point"""
import logging
import uvicorn
from fastapi import FastAPI

from service.api_routes import router as api_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)

# Create FastAPI application
app = FastAPI(
    title="Triton Proxy API",
    description="Vision service proxy for Triton"
)

# Register routers
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=4023)
