#!/usr/bin/env python3
"""
Rig Guardian Modules Entry Point
Registers all three modular services: YOLOv8 Detection, Pose Classification, Smart Camera
"""

import asyncio
import logging
from viam.module.module import Module

# Import all service modules
from yolov8_detection_service import YOLOv8DetectionService
from pose_classifier_service import PoseClassifierService  
from smart_camera_wrapper import SmartCameraWrapper

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

async def main():
    """Main entry point for all Rig Guardian modules"""
    try:
        LOGGER.info("🚀 Starting Rig Guardian Modular Services...")
        
        # Create module
        module = Module.from_args()
        
        # Register YOLOv8 Detection Service
        module.add_model_from_registry(
            YOLOv8DetectionService.MODEL, 
            YOLOv8DetectionService
        )
        LOGGER.info("✅ Registered YOLOv8 Detection Service")
        
        # Register Pose Classification Service
        module.add_model_from_registry(
            PoseClassifierService.MODEL,
            PoseClassifierService
        )
        LOGGER.info("✅ Registered Pose Classification Service")
        
        # Register Smart Camera Wrapper
        module.add_model_from_registry(
            SmartCameraWrapper.MODEL,
            SmartCameraWrapper
        )
        LOGGER.info("✅ Registered Smart Camera Wrapper")
        
        # Start module
        LOGGER.info("🎯 All services registered, starting module...")
        await module.start()
        
    except Exception as e:
        LOGGER.error(f"❌ Failed to start Rig Guardian modules: {e}")
        import traceback
        LOGGER.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
