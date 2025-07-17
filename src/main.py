import asyncio
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Starting module...")

from viam.module.module import Module
from viam.services.vision import Vision
logger.info("Imports successful, importing yolov8...")

try:
    from models.yolov8n_pose import Yolov8nPose
    logger.info("Yolov8nPose imported successfully.")
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.yolov8n_pose import Yolov8nPose
    logger.info("Yolov8nPose imported successfully from local module.")

async def main():
    """Main function to set up and run the module."""
    module = Module.from_args()
    logger.info(f"Starting module with address: {module.address}")
    
    # Register the vision service model
    module.add_model_from_registry(Vision.SUBTYPE, Yolov8nPose.MODEL)
    logger.info(f"Registered model: {Yolov8nPose.MODEL}")
    
    await module.start()

if __name__ == '__main__':
    asyncio.run(main())
