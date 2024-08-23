from fastapi import FastAPI

app = FastAPI()

# Import the routes
from .main import api_router

# Add routes to the app
app.include_router(api_router)
