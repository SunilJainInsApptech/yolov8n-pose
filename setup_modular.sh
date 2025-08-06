#!/bin/bash

echo "🚀 Setting up Rig Guardian Modular Architecture..."

# Create module directories
echo "📁 Creating module directories..."
mkdir -p modules/yolov8-detection
mkdir -p modules/pose-classifier  
mkdir -p modules/smart-camera

# Copy detection service
echo "📋 Setting up YOLOv8 Detection Service..."
cp yolov8_detection_service.py modules/yolov8-detection/main.py
cp requirements.txt modules/yolov8-detection/
echo "viam-sdk" > modules/yolov8-detection/requirements.txt

# Copy pose classifier service
echo "🧠 Setting up Pose Classification Service..."
cp pose_classifier_service.py modules/pose-classifier/main.py
cp fall_detection_alerts.py modules/pose-classifier/
cp pose_classifier.joblib modules/pose-classifier/
cp requirements.txt modules/pose-classifier/

# Copy smart camera wrapper
echo "📹 Setting up Smart Camera Wrapper..."
cp smart_camera_wrapper.py modules/smart-camera/main.py
echo "viam-sdk" > modules/smart-camera/requirements.txt

# Create meta.json for each module
echo "📝 Creating module metadata..."

# YOLOv8 Detection meta.json
cat > modules/yolov8-detection/meta.json << 'EOF'
{
  "module_id": "rig-guardian:yolov8n-detection",
  "visibility": "private",
  "url": "",
  "description": "YOLOv8 pose detection service for human pose detection",
  "models": [
    {
      "api": "rdk:service:vision",
      "model": "rig-guardian:yolov8n-pose:yolov8n-detection"
    }
  ],
  "entrypoint": "main.py"
}
EOF

# Pose Classifier meta.json
cat > modules/pose-classifier/meta.json << 'EOF'
{
  "module_id": "rig-guardian:pose-classifier",
  "visibility": "private", 
  "url": "",
  "description": "Pose classification service with fall detection and alerts",
  "models": [
    {
      "api": "rdk:service:generic",
      "model": "rig-guardian:pose-classifier:pose-classifier"
    }
  ],
  "entrypoint": "main.py"
}
EOF

# Smart Camera meta.json
cat > modules/smart-camera/meta.json << 'EOF'
{
  "module_id": "rig-guardian:smart-camera",
  "visibility": "private",
  "url": "", 
  "description": "Smart camera wrapper with configurable pose detection and fall alerts",
  "models": [
    {
      "api": "rdk:component:camera",
      "model": "rig-guardian:smart-camera:smart-camera"
    }
  ],
  "entrypoint": "main.py"
}
EOF

echo "✅ Modular architecture setup complete!"
echo ""
echo "📂 Module Structure:"
echo "modules/"
echo "├── yolov8-detection/    (YOLOv8 pose detection only)"
echo "├── pose-classifier/     (Pose classification + fall alerts)"
echo "└── smart-camera/        (Smart camera wrapper)"
echo ""
echo "🔧 Next steps:"
echo "1. Update viam_config_ml.json with the new module paths"
echo "2. Deploy each module separately to Viam Registry"
echo "3. Test each service independently"
echo "4. Configure per-camera settings in smart camera wrappers"
