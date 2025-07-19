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
import joblib
import numpy as np

LOGGER = getLogger(__name__)

MODEL_DIR = os.environ.get(
    "VIAM_MODULE_DATA", os.path.join(os.path.expanduser("~"), ".data", "models")
)

class Yolov8nPose(Vision, EasyResource):
    """
    Vision represents a Vision service.
    """
        
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("rig-guardian", "yolov8n-pose"), "yolov8n-pose"
    )

    MODEL_FILE = ""
    MODEL_REPO = ""
    MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, MODEL_REPO))

    model: YOLO
    pose_classifier: Optional[Any] = None
    device: str

    # Constructor
    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ComponentConfig):
        LOGGER.debug("Validating yolov8 service config")
        try:
            attrs = struct_to_dict(config.attributes)
            model_location = attrs.get("model_location")
            if not model_location:
                raise Exception("A model_location must be defined")
            LOGGER.debug(f"Validation successful for model_location: {model_location}")
            return []
        except Exception as e:
            LOGGER.error(f"Validation failed: {e}")
            raise e

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)
        model_location = str(attrs.get("model_location"))
        pose_classifier_path = attrs.get("pose_classifier_path")

        LOGGER.debug(f"Configuring yolov8 model with {model_location}")
        self.DEPS = dependencies
        self.task = str(attrs.get("task")) or None

        # Load pose classifier if specified
        if pose_classifier_path:
            try:
                import joblib
                classifier_path = os.path.abspath(pose_classifier_path)
                if os.path.exists(classifier_path):
                    self.pose_classifier = joblib.load(classifier_path)
                    LOGGER.info(f"Loaded ML pose classifier from {classifier_path}")
                else:
                    LOGGER.warning(f"Pose classifier file not found: {classifier_path}")
                    self.pose_classifier = None
            except ImportError:
                LOGGER.error("joblib not installed - cannot load ML pose classifier")
                self.pose_classifier = None
            except Exception as e:
                LOGGER.error(f"Failed to load pose classifier: {e}")
                self.pose_classifier = None
        else:
            LOGGER.info("No pose classifier specified - using rules-based classification")
            self.pose_classifier = None

        if "/" in model_location:
            if self.is_path(model_location):
                self.MODEL_PATH = model_location
            else:
                model_name = str(attrs.get("model_name", ""))
                if model_name == "":
                    raise Exception(
                        "model_name attribute is required for downloading models from HuggingFace."
                    )
                self.MODEL_REPO = model_location
                self.MODEL_FILE = model_name
                self.MODEL_PATH = os.path.abspath(
                    os.path.join(
                        MODEL_DIR,
                        f"{self.MODEL_REPO.replace('/', '_')}_{self.MODEL_FILE}",
                    )
                )

                self.get_model()

            self.model = YOLO(self.MODEL_PATH, task=self.task)
        else:
            self.model = YOLO(model_location, task=self.task)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        return

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return cam_image

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return await self.get_detections(await self.get_cam_image(camera_name))

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        detections = []
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
        
        if len(results) >= 1:
            result = results[0]
            
            # Debug logging
            LOGGER.info(f"Model result attributes: {dir(result)}")
            LOGGER.info(f"Has boxes: {hasattr(result, 'boxes')}")
            LOGGER.info(f"Has keypoints: {hasattr(result, 'keypoints')}")
            if hasattr(result, 'boxes') and result.boxes is not None:
                LOGGER.info(f"Number of boxes: {len(result.boxes)}")
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                LOGGER.info(f"Keypoints shape: {result.keypoints.xy.shape if result.keypoints.xy is not None else 'None'}")
            
            # Handle pose detection with both bounding boxes and keypoints
            if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result, 'keypoints') and result.keypoints is not None:
                LOGGER.info(f"Found {len(result.boxes)} people with keypoints")
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i]
                    confidence = result.boxes.conf[i].item()
                    class_id = int(result.boxes.cls[i].item())
                    class_name = result.names[class_id]
                    
                    # Extract keypoints for logging
                    keypoints = result.keypoints.xy[i].cpu().numpy()  # Shape: (17, 2) for COCO pose
                    visible_keypoints = sum(1 for x, y in keypoints if x > 0 and y > 0)
                    LOGGER.info(f"Person {i}: {visible_keypoints}/17 keypoints visible")
                    
                    # Create standard Detection object (no keypoints field)
                    detection = Detection(
                        x_min=int(box[0].item()),
                        y_min=int(box[1].item()),
                        x_max=int(box[2].item()),
                        y_max=int(box[3].item()),
                        confidence=confidence,
                        class_name=class_name
                    )
                    detections.append(detection)
                    
            # Fallback to regular bounding box detection if no keypoints
            elif hasattr(result, 'boxes') and result.boxes is not None:
                LOGGER.info(f"Found {len(result.boxes)} detections (no keypoints)")
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i]
                    confidence = result.boxes.conf[i].item()
                    class_id = int(result.boxes.cls[i].item())
                    class_name = result.names[class_id]
                    
                    detection = Detection(
                        x_min=int(box[0].item()),
                        y_min=int(box[1].item()),
                        x_max=int(box[2].item()),
                        y_max=int(box[3].item()),
                        confidence=confidence,
                        class_name=class_name
                    )
                    detections.append(detection)

        return detections

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return await self.get_classifications(await self.get_cam_image(camera_name))

    async def get_classifications(
        self,
        image: ViamImage,
        count: int = 0,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        classifications = []
        
        # Get pose classifications instead of trying to extract from YOLO classification output
        keypoints = await self.extract_keypoints(image)
        pose_classifications = await self.classify_poses(keypoints)
        
        # Convert pose classifications to Viam Classification format
        for pose_class in pose_classifications:
            if pose_class["pose_class"] != "unknown":
                classification = Classification(
                    class_name=pose_class["pose_class"],
                    confidence=pose_class["confidence"]
                )
                classifications.append(classification)
        
        return classifications

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        return []

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None
    ) -> Mapping[str, ValueTypes]:
        """Handle custom commands like getting keypoints or pose analysis."""
        cmd_name = command.get("command")
        
        if cmd_name == "get_keypoints":
            # Get keypoints from camera
            camera_name = command.get("camera_name", "")
            if camera_name:
                image = await self.get_cam_image(camera_name)
                keypoints = await self.extract_keypoints(image)
                return {"keypoints": keypoints}
            else:
                return {"error": "camera_name required"}
                
        elif cmd_name == "get_pose_analysis":
            # Get pose analysis (angles, distances, etc.)
            camera_name = command.get("camera_name", "")
            if camera_name:
                image = await self.get_cam_image(camera_name)
                analysis = await self.analyze_pose(image)
                return {"pose_analysis": analysis}
            else:
                return {"error": "camera_name required"}
                
        elif cmd_name == "get_pose_classifications":
            # Get detections + keypoints + pose classifications in one call
            camera_name = command.get("camera_name", "")
            if camera_name:
                image = await self.get_cam_image(camera_name)
                
                # Get both detections and keypoints
                detections = await self.get_detections(image)
                keypoints = await self.extract_keypoints(image)
                
                # Classify poses
                pose_classifications = await self.classify_poses(keypoints)
                
                return {
                    "detections": [
                        {
                            "x_min": det.x_min,
                            "y_min": det.y_min, 
                            "x_max": det.x_max,
                            "y_max": det.y_max,
                            "confidence": det.confidence,
                            "class_name": det.class_name
                        } for det in detections
                    ],
                    "keypoints": keypoints,
                    "pose_classifications": pose_classifications
                }
            else:
                return {"error": "camera_name required"}
        
        return {"supported_commands": ["get_keypoints", "get_pose_analysis", "get_pose_classifications"]}

    async def extract_keypoints(self, image: ViamImage) -> List[dict]:
        """Extract only keypoints from detected persons."""
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
        
        all_keypoints = []
        if len(results) >= 1:
            result = results[0]
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoint_names = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]
                
                for person_idx in range(len(result.keypoints)):
                    keypoints = result.keypoints.xy[person_idx].cpu().numpy()
                    keypoint_conf = result.keypoints.conf[person_idx].cpu().numpy() if result.keypoints.conf is not None else None
                    
                    person_keypoints = {
                        "person_id": person_idx,
                        "keypoints": []
                    }
                    
                    for j, (x, y) in enumerate(keypoints):
                        keypoint_data = {
                            "name": keypoint_names[j] if j < len(keypoint_names) else f"keypoint_{j}",
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(keypoint_conf[j]) if keypoint_conf is not None else 1.0,
                            "visible": float(x) > 0 and float(y) > 0
                        }
                        person_keypoints["keypoints"].append(keypoint_data)
                    
                    all_keypoints.append(person_keypoints)
        
        return all_keypoints

    async def analyze_pose(self, image: ViamImage) -> List[dict]:
        """Analyze pose for basic metrics like angles and distances."""
        keypoints_data = await self.extract_keypoints(image)
        
        analyses = []
        for person_data in keypoints_data:
            keypoints = {kp["name"]: (kp["x"], kp["y"]) for kp in person_data["keypoints"] if kp["visible"]}
            
            analysis = {
                "person_id": person_data["person_id"],
                "pose_metrics": {}
            }
            
            # Calculate arm angles if shoulder, elbow, wrist are visible
            if all(joint in keypoints for joint in ["left_shoulder", "left_elbow", "left_wrist"]):
                left_arm_angle = self.calculate_angle(
                    keypoints["left_shoulder"], 
                    keypoints["left_elbow"], 
                    keypoints["left_wrist"]
                )
                analysis["pose_metrics"]["left_arm_angle"] = left_arm_angle
            
            if all(joint in keypoints for joint in ["right_shoulder", "right_elbow", "right_wrist"]):
                right_arm_angle = self.calculate_angle(
                    keypoints["right_shoulder"], 
                    keypoints["right_elbow"], 
                    keypoints["right_wrist"]
                )
                analysis["pose_metrics"]["right_arm_angle"] = right_arm_angle
            
            # Calculate body posture
            if all(joint in keypoints for joint in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                shoulder_width = abs(keypoints["left_shoulder"][0] - keypoints["right_shoulder"][0])
                hip_width = abs(keypoints["left_hip"][0] - keypoints["right_hip"][0])
                analysis["pose_metrics"]["shoulder_width"] = float(shoulder_width)
                analysis["pose_metrics"]["hip_width"] = float(hip_width)
            
            analyses.append(analysis)
        
        return analyses

    async def classify_poses(self, keypoints_data: List[dict]) -> List[dict]:
        """Classify poses as standing, sitting, crouching, or fallen based on keypoints."""
        classifications = []
        
        for person_data in keypoints_data:
            keypoints = {kp["name"]: (kp["x"], kp["y"]) for kp in person_data["keypoints"] if kp["visible"]}
            
            classification = {
                "person_id": person_data["person_id"],
                "pose_class": "unknown",
                "confidence": 0.0,
                "reasoning": [],
                "method": "ml" if self.pose_classifier else "rules_based"
            }
            
            # Check if we have enough keypoints for classification
            required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            missing_points = [point for point in required_points if point not in keypoints]
            
            if missing_points:
                classification["reasoning"].append(f"Missing keypoints: {missing_points}")
                classifications.append(classification)
                continue
            
            # Use ML classifier if available
            if self.pose_classifier:
                try:
                    features = self.extract_pose_features_for_ml(keypoints)
                    if features is not None:
                        import numpy as np
                        features_array = np.array(features).reshape(1, -1)
                        prediction = self.pose_classifier.predict(features_array)[0]
                        probabilities = self.pose_classifier.predict_proba(features_array)[0]
                        
                        # Get class names from classifier
                        class_names = self.pose_classifier.classes_
                        confidence = max(probabilities)
                        
                        classification["pose_class"] = prediction
                        classification["confidence"] = float(confidence)
                        classification["reasoning"].append(f"ML prediction with {len(features)} features")
                        
                        # Add probability distribution for debugging
                        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                        classification["probabilities"] = prob_dict
                        
                        LOGGER.info(f"ML Classification for person {person_data['person_id']}: {prediction} ({confidence:.3f})")
                        LOGGER.info(f"Probabilities: {prob_dict}")
                        
                        classifications.append(classification)
                        continue
                    else:
                        classification["reasoning"].append("Failed to extract ML features - falling back to rules")
                except Exception as e:
                    LOGGER.error(f"ML classification failed: {e}")
                    classification["reasoning"].append(f"ML classification error: {e}")
            
            # Fallback to rules-based classification
            classification["method"] = "rules_based"
            classification = await self.classify_pose_rules_based(keypoints, classification)
            classifications.append(classification)
        
        return classifications

    def extract_pose_features_for_ml(self, keypoints: dict) -> Optional[List[float]]:
        """Extract features for ML classifier (matches training format)."""
        try:
            features = []
            
            # Check required points
            required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            if not all(point in keypoints for point in required_points):
                return None
            
            # Basic body measurements
            head_y = keypoints["nose"][1]
            shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
            hip_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
            
            # Vertical distances (key for pose classification)
            head_to_shoulder = head_y - shoulder_y
            shoulder_to_hip = shoulder_y - hip_y
            head_to_hip = head_y - hip_y
            
            # Body proportions
            shoulder_width = abs(keypoints["left_shoulder"][0] - keypoints["right_shoulder"][0])
            hip_width = abs(keypoints["left_hip"][0] - keypoints["right_hip"][0])
            
            features.extend([
                head_to_shoulder,    # Negative if head above shoulders (normal)
                shoulder_to_hip,     # Negative if shoulders above hips (normal)
                head_to_hip,         # Overall body height
                shoulder_width,      # Body width at shoulders
                hip_width,           # Body width at hips
            ])
            
            # Knee positions if available
            if "left_knee" in keypoints and "right_knee" in keypoints:
                knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
                hip_to_knee = hip_y - knee_y
                features.extend([
                    hip_to_knee,     # Negative if hips above knees (normal standing)
                    knee_y           # Absolute knee position
                ])
            else:
                features.extend([0, 0])
                
            # Ankle positions if available
            if "left_ankle" in keypoints and "right_ankle" in keypoints:
                ankle_y = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2
                if "left_knee" in keypoints and "right_knee" in keypoints:
                    knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
                    knee_to_ankle = knee_y - ankle_y
                    features.extend([knee_to_ankle, ankle_y])
                else:
                    features.extend([0, ankle_y])
            else:
                features.extend([0, 0])
                
            # Body aspect ratio
            if shoulder_to_hip != 0:
                features.append(abs(head_to_hip) / abs(shoulder_to_hip))
            else:
                features.append(0)
                
            # Normalized positions (relative to body height)
            body_height = abs(head_to_hip) if abs(head_to_hip) > 0 else 1
            features.extend([
                head_y / body_height,
                shoulder_y / body_height, 
                hip_y / body_height
            ])
            
            return features
            
        except Exception as e:
            LOGGER.error(f"Feature extraction failed: {e}")
            return None

    async def classify_pose_rules_based(self, keypoints: dict, classification: dict) -> dict:
        """Rules-based pose classification (fallback method)."""
        # Calculate body metrics
        head_y = keypoints["nose"][1]
        shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
        hip_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
        
        # Debug logging for pose classification
        LOGGER.info(f"Person {classification['person_id']} pose metrics:")
        LOGGER.info(f"  head_y: {head_y}, shoulder_y: {shoulder_y}, hip_y: {hip_y}")
        
        # Get knee and ankle positions if available
        knee_y = None
        ankle_y = None
        
        if "left_knee" in keypoints and "right_knee" in keypoints:
            knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
            LOGGER.info(f"  knee_y: {knee_y}")
        if "left_ankle" in keypoints and "right_ankle" in keypoints:
            ankle_y = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2
            LOGGER.info(f"  ankle_y: {ankle_y}")
        
        # Calculate body orientation (vertical distance ratios)
        head_to_shoulder = abs(head_y - shoulder_y)
        shoulder_to_hip = abs(shoulder_y - hip_y)
        
        LOGGER.info(f"  head_to_shoulder: {head_to_shoulder}, shoulder_to_hip: {shoulder_to_hip}")
        
        # Rule-based classification
        score_standing = 0
        score_sitting = 0  
        score_crouching = 0
        score_fallen = 0
        
        # FALLEN: Check if person is horizontal (head and hip at similar y-level)
        if abs(head_y - hip_y) < shoulder_to_hip * 0.5:
            score_fallen += 3
            classification["reasoning"].append("Head and hip at similar level (horizontal)")
        
        # STANDING: Head significantly above hips, knees below hips
        if head_y < hip_y - shoulder_to_hip * 1.5:  # Head well above hips
            score_standing += 2
            classification["reasoning"].append("Head well above hips")
            
            if knee_y and knee_y > hip_y:  # Knees below hips
                score_standing += 2
                classification["reasoning"].append("Knees below hips")
                
            if ankle_y and ankle_y > knee_y:  # Ankles below knees
                score_standing += 1
                classification["reasoning"].append("Ankles below knees")
        
        # SITTING: Hip and knee at similar level, head above hips
        if knee_y and abs(hip_y - knee_y) < shoulder_to_hip * 0.8:
            score_sitting += 2
            classification["reasoning"].append("Hips and knees at similar level")
            
            if head_y < hip_y:  # Head above hips
                score_sitting += 1
                classification["reasoning"].append("Head above hips")
        
        # CROUCHING: Knees significantly above hips, but head still visible
        if knee_y and knee_y < hip_y - shoulder_to_hip * 0.5:
            score_crouching += 2
            classification["reasoning"].append("Knees above hips")
            
            if head_y < shoulder_y:  # Head above shoulders
                score_crouching += 1
                classification["reasoning"].append("Head above shoulders")
        
        # Determine final classification
        scores = {
            "standing": score_standing,
            "sitting": score_sitting,
            "crouching": score_crouching,
            "fallen": score_fallen
        }
        
        # Debug logging for scores
        LOGGER.info(f"  Classification scores: {scores}")
        LOGGER.info(f"  Reasoning so far: {classification['reasoning']}")
        
        max_score = max(scores.values())
        if max_score > 0:
            pose_class = max(scores.keys(), key=lambda k: scores[k])
            classification["pose_class"] = pose_class
            classification["confidence"] = min(max_score / 5.0, 1.0)  # Normalize to 0-1
            LOGGER.info(f"  Final classification: {pose_class} (confidence: {classification['confidence']})")
        else:
            classification["pose_class"] = "unknown"
            classification["confidence"] = 0.0
            classification["reasoning"].append("Insufficient evidence for any pose class")
            LOGGER.info(f"  Final classification: unknown")
        
        # Add raw metrics for debugging
        classification["metrics"] = {
            "head_y": float(head_y),
            "shoulder_y": float(shoulder_y), 
            "hip_y": float(hip_y),
            "knee_y": float(knee_y) if knee_y else None,
            "ankle_y": float(ankle_y) if ankle_y else None,
            "scores": scores
        }
        
        return classification

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)."""
        import math
        
        # Vector from point2 to point1
        v1 = (point1[0] - point2[0], point1[1] - point2[1])
        # Vector from point2 to point3  
        v2 = (point3[0] - point2[0], point3[1] - point2[1])
        
        # Calculate dot product and magnitudes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0
        
        # Calculate angle in radians, then convert to degrees
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return float(angle_deg)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)
        result.detections = await self.get_detections(result.image)
        result.classifications = await self.get_classifications(result.image, 1)
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    def is_path(self, path: str) -> bool:
        try:
            Path(path)
            return os.path.exists(path)
        except ValueError:
            return False

    def get_model(self):
        if not os.path.exists(self.MODEL_PATH):
            MODEL_URL = f"https://huggingface.co/{self.MODEL_REPO}/resolve/main/{self.MODEL_FILE}"
            LOGGER.debug(f"Fetching model {self.MODEL_FILE} from {MODEL_URL}")
            urlretrieve(MODEL_URL, self.MODEL_PATH, self.log_progress)

    def log_progress(self, count: int, block_size: int, total_size: int) -> None:
        percent = count * block_size * 100 // total_size
        LOGGER.debug(f"\rDownloading {self.MODEL_FILE}: {percent}%")


# vendored and updated from ultralyticsplus library
def postprocess_classify_output(model: YOLO, result: Results) -> dict:
    """
    Postprocesses the output of classification models

    Args:
        model (YOLO): YOLO model
        prob (np.ndarray): output of the model

    Returns:
        dict: dictionary of outputs with labels
    """
    output = {}
    if isinstance(model.names, list):
        names = model.names
    elif isinstance(model.names, dict):
        names = model.names.values()
    else:
        raise ValueError("Model names must be either a list or a dict")

    if result.probs:
        for i, label in enumerate(names):
            output[label] = result.probs[i].item()
        return output
    else:
        return {}