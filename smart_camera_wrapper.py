#!/usr/bin/env python3
"""
Smart Camera Wrapper Module
Wraps RTSP cameras with configurable pose detection and fall alerts
"""

import asyncio
import logging
import base64
from typing import Dict, Any, Optional, List
from viam.module.module import Module
from viam.components.camera import Camera
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict
from viam.media.video import ViamImage

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class SmartCameraWrapper(Camera):
    """
    Smart Camera Wrapper that adds pose detection and fall alert capabilities
    to regular RTSP cameras
    """
    
    MODEL: Model = Model(ModelFamily("rig-guardian", "smart-camera"), "smart-camera")
    
    def __init__(self, name: str):
        super().__init__(name)
        self.rtsp_camera = None
        self.yolo_detector = None
        self.pose_classifier_service = None
        self.camera_config = {
            "location": "Unknown",
            "enable_pose_classification": True,
            "enable_fall_alerts": True,
            "fall_confidence_threshold": 0.7,
            "detection_confidence_threshold": 0.5
        }
        
    @classmethod
    def new(cls, config, dependencies):
        """Create new smart camera wrapper"""
        camera = cls(config.name)
        camera.reconfigure(config, dependencies)
        return camera
    
    @classmethod  
    def validate_config(cls, config) -> None:
        """Validate configuration"""
        # Add validation logic here
        pass
    
    def reconfigure(self, config, dependencies):
        """Configure the smart camera wrapper"""
        try:
            attributes = struct_to_dict(config.attributes)
            
            # Update camera configuration
            self.camera_config.update({
                "location": attributes.get("location", "Unknown"),
                "enable_pose_classification": attributes.get("enable_pose_classification", True),
                "enable_fall_alerts": attributes.get("enable_fall_alerts", True),
                "fall_confidence_threshold": attributes.get("fall_confidence_threshold", 0.7),
                "detection_confidence_threshold": attributes.get("detection_confidence_threshold", 0.5)
            })
            
            # Get the underlying RTSP camera
            rtsp_camera_name = attributes.get("rtsp_camera_name")
            if rtsp_camera_name and dependencies:
                for dep in dependencies:
                    if dep.name == rtsp_camera_name:
                        self.rtsp_camera = dep
                        break
            
            # Get YOLOv8 detector service
            yolo_service_name = attributes.get("yolo_service_name", "yolo-detector")
            if yolo_service_name and dependencies:
                for dep in dependencies:
                    if dep.name == yolo_service_name:
                        self.yolo_detector = dep
                        break
            
            # Get pose classifier service
            pose_service_name = attributes.get("pose_service_name", "pose-classifier-service")
            if pose_service_name and dependencies:
                for dep in dependencies:
                    if dep.name == pose_service_name:
                        self.pose_classifier_service = dep
                        break
            
            LOGGER.info(f"✅ Smart camera {config.name} configured:")
            LOGGER.info(f"   Location: {self.camera_config['location']}")
            LOGGER.info(f"   Pose Classification: {self.camera_config['enable_pose_classification']}")
            LOGGER.info(f"   Fall Alerts: {self.camera_config['enable_fall_alerts']}")
            
            # Register this camera with the pose classifier service
            if self.pose_classifier_service and self.camera_config['enable_pose_classification']:
                asyncio.create_task(self._register_with_pose_service())
                
        except Exception as e:
            LOGGER.error(f"❌ Failed to configure smart camera {config.name}: {e}")
            raise
    
    async def _register_with_pose_service(self):
        """Register this camera with the pose classifier service"""
        try:
            await self.pose_classifier_service.do_command({
                "command": "register_camera",
                "camera_name": self.name,
                "config": self.camera_config
            })
            LOGGER.info(f"✅ Camera {self.name} registered with pose classifier service")
        except Exception as e:
            LOGGER.error(f"❌ Failed to register camera {self.name} with pose service: {e}")
    
    async def get_image(self, mime_type: str = "", *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> ViamImage:
        """Get processed image with optional pose detection overlay"""
        if not self.rtsp_camera:
            raise Exception("No RTSP camera configured")
        
        # Get original image from RTSP camera
        image = await self.rtsp_camera.get_image(mime_type, extra=extra, timeout=timeout)
        
        # If pose classification is disabled, return original image
        if not self.camera_config.get("enable_pose_classification", True):
            return image
        
        # Process with YOLOv8 detection and pose classification
        try:
            processed_image = await self._process_image_with_detection(image)
            return processed_image
        except Exception as e:
            LOGGER.error(f"❌ Image processing failed for {self.name}: {e}")
            # Return original image if processing fails
            return image
    
    async def _process_image_with_detection(self, image: ViamImage) -> ViamImage:
        """Process image with YOLOv8 detection and pose classification"""
        try:
            # Step 1: Run YOLOv8 detection
            if not self.yolo_detector:
                return image
            
            detection_result = await self.yolo_detector.get_detections(image)
            detections = detection_result.detections if hasattr(detection_result, 'detections') else []
            
            # Step 2: Run pose classification if enabled and poses detected
            pose_results = None
            if (self.camera_config.get("enable_pose_classification", True) and 
                self.pose_classifier_service and detections):
                
                # Convert image to base64 for pose service
                image_data = base64.b64encode(image.data).decode('utf-8')
                
                # Format detections for pose service
                formatted_detections = []
                for i, det in enumerate(detections):
                    detection_dict = {
                        "id": f"det_{i}",
                        "bbox": [det.x_min, det.y_min, det.x_max, det.y_max],
                        "confidence": det.confidence,
                        "keypoints": getattr(det, 'keypoints', [])
                    }
                    formatted_detections.append(detection_dict)
                
                # Send to pose classifier
                pose_response = await self.pose_classifier_service.do_command({
                    "command": "classify_poses",
                    "camera_name": self.name,
                    "detections": formatted_detections,
                    "image_data": image_data
                })
                
                pose_results = pose_response.get("classified_poses", [])
                
                if pose_response.get("fall_detected", False):
                    LOGGER.warning(f"🚨 Fall detected on camera {self.name}!")
            
            # Step 3: Draw overlays on image (optional)
            overlay_image = self._draw_detection_overlays(image, detections, pose_results)
            
            return overlay_image
            
        except Exception as e:
            LOGGER.error(f"❌ Detection processing error: {e}")
            return image
    
    def _draw_detection_overlays(self, image: ViamImage, detections: List, pose_results: Optional[List] = None) -> ViamImage:
        """Draw detection boxes and pose information on image"""
        try:
            # For now, return original image
            # You can implement overlay drawing here using PIL or OpenCV
            return image
            
        except Exception as e:
            LOGGER.error(f"❌ Overlay drawing error: {e}")
            return image
    
    async def get_properties(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Camera.Properties:
        """Get camera properties from underlying RTSP camera"""
        if self.rtsp_camera:
            return await self.rtsp_camera.get_properties(extra=extra, timeout=timeout)
        else:
            # Return default properties
            return Camera.Properties(
                supports_pcd=False,
                intrinsic_parameters=None,
                distortion_parameters=None,
                mime_types=["image/jpeg"]
            )
    
    async def get_point_cloud(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
        """Point cloud not supported"""
        raise NotImplementedError("Point cloud not supported by smart camera wrapper")
    
    async def do_command(self, command: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Handle custom commands"""
        cmd = command.get("command")
        
        if cmd == "get_config":
            return {"config": self.camera_config}
        elif cmd == "update_config":
            new_config = command.get("config", {})
            self.camera_config.update(new_config)
            return {"success": True, "updated_config": self.camera_config}
        elif cmd == "get_stats":
            return await self._get_camera_stats()
        else:
            # Forward unknown commands to underlying camera
            if self.rtsp_camera:
                return await self.rtsp_camera.do_command(command, timeout=timeout)
            else:
                return {"error": f"Unknown command: {cmd}"}
    
    async def _get_camera_stats(self) -> Dict[str, Any]:
        """Get camera statistics"""
        return {
            "camera_name": self.name,
            "location": self.camera_config["location"],
            "pose_classification_enabled": self.camera_config["enable_pose_classification"],
            "fall_alerts_enabled": self.camera_config["enable_fall_alerts"],
            "fall_threshold": self.camera_config["fall_confidence_threshold"],
            "detection_threshold": self.camera_config["detection_confidence_threshold"],
            "rtsp_camera_connected": self.rtsp_camera is not None,
            "yolo_detector_connected": self.yolo_detector is not None,
            "pose_service_connected": self.pose_classifier_service is not None
        }

# Module registration
async def main():
    """Main entry point for the smart camera module"""
    module = Module.from_args()
    module.add_model_from_registry(SmartCameraWrapper.MODEL, SmartCameraWrapper)
    await module.start()

if __name__ == "__main__":
    asyncio.run(main())
