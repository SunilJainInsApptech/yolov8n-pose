#!/usr/bin/env python3
"""
YOLOv8 Detection Service (Detection Only)
Focused only on YOLOv8 pose detection without classification or alerts
"""

import os
from pathlib import Path
from typing import ClassVar, Mapping, Any, Optional, List, cast
from typing_extensions import Self
from urllib.request import urlretrieve

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.utils import ValueTypes

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.logging import getLogger
from viam.utils import struct_to_dict

from ultralytics.engine.results import Results
from ultralytics import YOLO
import torch
import numpy as np

LOGGER = getLogger(__name__)

MODEL_DIR = os.environ.get(
    "VIAM_MODULE_DATA", os.path.join(os.path.expanduser("~"), ".data", "models")
)

class YOLOv8DetectionService(Vision, EasyResource):
    """
    YOLOv8 Detection Service - Pure detection without pose classification
    """
        
    MODEL: ClassVar[Model] = Model(
        ModelFamily("rig-guardian", "yolov8n-pose"), "yolov8n-detection"
    )

    MODEL_FILE = ""
    MODEL_REPO = ""
    MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, MODEL_REPO))

    model: YOLO
    device: str
    confidence_threshold: float = 0.5

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """Create new YOLOv8 detection service instance"""
        LOGGER.info("Creating YOLOv8 detection service instance")
        try:
            result = super().new(config, dependencies)
            LOGGER.info("YOLOv8 detection service created successfully")
            return result
        except Exception as e:
            LOGGER.error(f"Failed to create YOLOv8 detection service: {e}")
            raise

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        """Validate configuration"""
        LOGGER.info("Validating YOLOv8 detection service config")
        try:
            attrs = struct_to_dict(config.attributes)
            model_location = attrs.get("model_location")
            if not model_location:
                raise Exception("A model_location must be defined")
            LOGGER.info(f"Config validation successful for model: {model_location}")
            return []
        except Exception as e:
            LOGGER.error(f"Config validation failed: {e}")
            raise e

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Configure the YOLOv8 detection service"""
        LOGGER.info("Reconfiguring YOLOv8 detection service")
        
        attrs = struct_to_dict(config.attributes)
        model_location = str(attrs.get("model_location"))
        self.confidence_threshold = float(attrs.get("confidence_threshold", 0.5))
        
        LOGGER.info(f"Model location: {model_location}")
        LOGGER.info(f"Confidence threshold: {self.confidence_threshold}")
        
        self.DEPS = dependencies
        self.task = str(attrs.get("task")) or None

        # Set up device
        if torch.cuda.is_available():
            self.device = "cuda"
            LOGGER.info("✅ CUDA detected - using GPU acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            LOGGER.info("✅ MPS detected - using Apple Silicon acceleration")
        else:
            self.device = "cpu"
            LOGGER.info("⚠️ Using CPU - consider GPU for better performance")

        # Load YOLOv8 model
        try:
            if os.path.exists(model_location):
                LOGGER.info(f"Loading model from local path: {model_location}")
                self.model = YOLO(model_location)
            else:
                LOGGER.info(f"Downloading model from Ultralytics Hub: {model_location}")
                self.model = YOLO(model_location)
            
            # Move model to device
            self.model.to(self.device)
            LOGGER.info(f"✅ YOLOv8 model loaded successfully on {self.device}")
            
        except Exception as e:
            LOGGER.error(f"❌ Failed to load YOLOv8 model: {e}")
            raise Exception(f"Model loading failed: {e}")

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Get pose detections from YOLOv8"""
        try:
            # Convert Viam image to PIL
            pil_image = viam_to_pil_image(image)
            
            # Run YOLOv8 detection
            results = self.model(pil_image, conf=self.confidence_threshold, device=self.device)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    # Get keypoints if available (YOLOv8-pose)
                    keypoints = None
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.cpu().numpy()
                    
                    for i, box in enumerate(boxes.data):
                        # Parse bounding box
                        x1, y1, x2, y2, conf, cls = box[:6]
                        
                        # Get keypoints for this detection if available
                        detection_keypoints = []
                        if keypoints is not None and i < len(keypoints.data):
                            kpts = keypoints.data[i]  # Shape: (17, 3) for COCO pose
                            detection_keypoints = kpts.tolist()
                        
                        # Create detection object
                        detection = Detection(
                            x_min=int(x1),
                            y_min=int(y1),
                            x_max=int(x2),
                            y_max=int(y2),
                            confidence=float(conf),
                            class_name=self.model.names[int(cls)]
                        )
                        
                        # Store keypoints in detection if available
                        if detection_keypoints:
                            # Add keypoints as extra metadata
                            setattr(detection, 'keypoints', detection_keypoints)
                        
                        detections.append(detection)
            
            LOGGER.info(f"✅ Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            LOGGER.error(f"❌ Detection failed: {e}")
            return []

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Get detections from a specific camera"""
        try:
            # Get camera from dependencies
            camera = None
            for dep in self.DEPS.values():
                if hasattr(dep, 'name') and dep.name == camera_name:
                    camera = dep
                    break
            
            if not camera:
                raise Exception(f"Camera '{camera_name}' not found in dependencies")
            
            # Get image from camera
            image = await camera.get_image()
            
            # Run detection
            return await self.get_detections(image, extra=extra, timeout=timeout)
            
        except Exception as e:
            LOGGER.error(f"❌ Camera detection failed for {camera_name}: {e}")
            return []

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Classifications not supported - this is detection only"""
        return []

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Classifications not supported - this is detection only"""
        return []

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        """Point clouds not supported"""
        return []

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = True,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        """Capture all data from camera"""
        result = CaptureAllResult()
        
        try:
            # Get camera
            camera = None
            for dep in self.DEPS.values():
                if hasattr(dep, 'name') and dep.name == camera_name:
                    camera = dep
                    break
            
            if not camera:
                raise Exception(f"Camera '{camera_name}' not found")
            
            # Get image
            image = await camera.get_image()
            
            if return_image:
                result.image = image
            
            if return_detections:
                result.detections = await self.get_detections(image, extra=extra, timeout=timeout)
            
            # Classifications and point clouds not supported
            result.classifications = []
            result.objects = []
            
        except Exception as e:
            LOGGER.error(f"❌ Capture all failed for {camera_name}: {e}")
        
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        """Get vision service properties"""
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
    ) -> Mapping[str, ValueTypes]:
        """Handle custom commands"""
        cmd = command.get("command")
        
        if cmd == "get_model_info":
            return {
                "model_type": "YOLOv8-pose",
                "device": self.device,
                "confidence_threshold": self.confidence_threshold,
                "model_path": getattr(self.model, 'model_path', 'unknown')
            }
        elif cmd == "set_confidence":
            new_conf = float(command.get("confidence", 0.5))
            self.confidence_threshold = max(0.0, min(1.0, new_conf))
            return {"confidence_threshold": self.confidence_threshold}
        else:
            return {"error": f"Unknown command: {cmd}"}
