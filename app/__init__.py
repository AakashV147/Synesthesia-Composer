from fastapi import FastAPI
import logging
from app.main import api_router  # Correctly import the router instance

# Initialize logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("App module initialized")

app = FastAPI()

# Add routes to the app
app.include_router(api_router)  # Include the router instance correctly
