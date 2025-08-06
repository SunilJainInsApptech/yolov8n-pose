# PowerShell script for setting up modular architecture
Write-Host "Setting up Rig Guardian Modular Architecture..." -ForegroundColor Green

# Create module directories
Write-Host "Creating module directories..." -ForegroundColor Yellow
New-Item -Path "modules/yolov8-detection" -ItemType Directory -Force | Out-Null
New-Item -Path "modules/pose-classifier" -ItemType Directory -Force | Out-Null  
New-Item -Path "modules/smart-camera" -ItemType Directory -Force | Out-Null

# Copy detection service
Write-Host "Setting up YOLOv8 Detection Service..." -ForegroundColor Yellow
Copy-Item "yolov8_detection_service.py" "modules/yolov8-detection/main.py" -Force
Copy-Item "requirements.txt" "modules/yolov8-detection/" -Force

# Copy pose classifier service  
Write-Host "Setting up Pose Classification Service..." -ForegroundColor Yellow
Copy-Item "pose_classifier_service.py" "modules/pose-classifier/main.py" -Force
Copy-Item "src/fall_detection_alerts.py" "modules/pose-classifier/" -Force -ErrorAction SilentlyContinue
Copy-Item "pose_classifier.joblib" "modules/pose-classifier/" -Force -ErrorAction SilentlyContinue
Copy-Item "requirements.txt" "modules/pose-classifier/" -Force

# Copy smart camera wrapper
Write-Host "Setting up Smart Camera Wrapper..." -ForegroundColor Yellow
Copy-Item "smart_camera_wrapper.py" "modules/smart-camera/main.py" -Force
"viam-sdk" | Out-File "modules/smart-camera/requirements.txt" -Encoding UTF8

# Create meta.json for YOLOv8 Detection
Write-Host "Creating YOLOv8 Detection metadata..." -ForegroundColor Yellow
@'
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
'@ | Out-File "modules/yolov8-detection/meta.json" -Encoding UTF8

# Create meta.json for Pose Classifier
Write-Host "Creating Pose Classifier metadata..." -ForegroundColor Yellow
@'
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
'@ | Out-File "modules/pose-classifier/meta.json" -Encoding UTF8

# Create meta.json for Smart Camera
Write-Host "Creating Smart Camera metadata..." -ForegroundColor Yellow
@'
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
'@ | Out-File "modules/smart-camera/meta.json" -Encoding UTF8

Write-Host "Modular architecture setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Module Structure:" -ForegroundColor Cyan
Write-Host "modules/"
Write-Host "├── yolov8-detection/    (YOLOv8 pose detection only)"
Write-Host "├── pose-classifier/     (Pose classification + fall alerts)"
Write-Host "└── smart-camera/        (Smart camera wrapper)"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Update viam_config_ml.json with the new module paths"
Write-Host "2. Deploy each module separately to Viam Registry"
Write-Host "3. Test each service independently"
Write-Host "4. Configure per-camera settings in smart camera wrappers"
