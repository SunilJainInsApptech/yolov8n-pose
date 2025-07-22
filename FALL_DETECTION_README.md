# Fall Detection Alert System

This YOLOv8 pose detection module now includes an integrated fall detection alert system that automatically sends SMS notifications via Twilio when a person is detected as fallen.

## Features

- 🤖 **ML-based fall detection** using RandomForest classifier trained on pose features
- 📱 **SMS alerts via Twilio** with customizable message content
- 📸 **Image capture** of fall events for evidence/verification
- ⏰ **Alert cooldown** to prevent spam (configurable)
- 🎯 **Confidence thresholds** to reduce false positives
- 📊 **Detailed metadata** in alerts (pose probabilities, features, etc.)

## Quick Setup

### 1. Install Dependencies

The Twilio library is already added to `requirements.txt`. If you need to install manually:

```bash
pip install twilio
```

### 2. Get Twilio Credentials

1. Sign up at [Twilio](https://www.twilio.com/)
2. Get your **Account SID** and **Auth Token** from the Console
3. Purchase a **Twilio phone number** for sending SMS
4. Note the phone numbers you want to receive alerts

### 3. Update Robot Configuration (Secure Method)

**🔒 RECOMMENDED: Use Environment Variables for Security**

Set up environment variables on your system:

```bash
# Add to ~/.bashrc or ~/.profile
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token_here"
export TWILIO_FROM_PHONE="+15551234567" 
export TWILIO_TO_PHONES="+15557654321,+15559876543"

# Reload environment
source ~/.bashrc
```

Then use this secure robot configuration:

```json
{
  "name": "yolov8-pose-detector", 
  "type": "vision",
  "model": "rig-guardian:yolov8n-pose:yolov8n-pose",
  "attributes": {
    "model_location": "/home/sunil/yolov8n-pose.pt",
    "pose_classifier_path": "/home/sunil/yolov8n-pose/pose_classifier.joblib",
    
    "use_env_for_twilio": true,
    "fall_confidence_threshold": 0.7,
    "alert_cooldown_seconds": 300
  }
}
```

**⚠️ Alternative: Direct Configuration (Less Secure)**

If you can't use environment variables, you can put credentials directly in config (NOT recommended for production):

```json
{
  "name": "yolov8-pose-detector",
  "type": "vision", 
  "model": "rig-guardian:yolov8n-pose:yolov8n-pose",
  "attributes": {
    "model_location": "/home/sunil/yolov8n-pose.pt",
    "pose_classifier_path": "/home/sunil/yolov8n-pose/pose_classifier.joblib",
    
    "twilio_account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "twilio_auth_token": "your_auth_token_here",
    "twilio_from_phone": "+15551234567",
    "twilio_to_phones": ["+15557654321", "+15559876543"],
    
    "fall_confidence_threshold": 0.7,
    "alert_cooldown_seconds": 300
  }
}
```

### 4. Test the Setup

```bash
python test_fall_alerts.py
```

## Configuration Options

| Setting | Description | Default | Example |
|---------|-------------|---------|---------|
| `twilio_account_sid` | Your Twilio Account SID | Required | `"ACxxxxxxxxx..."` |
| `twilio_auth_token` | Your Twilio Auth Token | Required | `"your_token..."` |
| `twilio_from_phone` | Twilio phone number for sending | Required | `"+15551234567"` |
| `twilio_to_phones` | List of recipient phone numbers | Required | `["+15557654321"]` |
| `fall_confidence_threshold` | Minimum confidence to trigger alert | `0.7` | `0.8` |
| `alert_cooldown_seconds` | Seconds between alerts per person | `300` | `600` |
| `webhook_url` | URL for image uploads (optional) | `None` | `"https://..."` |

## How It Works

1. **Pose Detection**: YOLOv8 detects people and their pose keypoints
2. **ML Classification**: RandomForest classifier analyzes pose features to classify as standing/sitting/crouching/fallen
3. **Fall Detection**: When pose is classified as "fallen" with confidence above threshold
4. **Alert Trigger**: System captures image and sends SMS alert with details
5. **Cooldown**: Prevents duplicate alerts for same person within cooldown period

## Alert Message Format

```
🚨 FALL DETECTED 🚨
Camera: living_room_camera
Person: person_1
Confidence: 85.0%
Time: 2025-01-20 14:30:15
Pose Probs: Fall:85.0% Stand:10.0% Sit:3.0%
Image: fall_detection_person_1_20250120_143015.jpg

Please check the location immediately.
```

## Troubleshooting

### No alerts being sent
- Check Twilio credentials are correct
- Verify phone numbers include country code (e.g., +1 for US)
- Check logs for "FALL DETECTION ALERTS INITIALIZED SUCCESSFULLY"
- Ensure ML classifier is loaded properly

### False positives
- Increase `fall_confidence_threshold` (try 0.8 or 0.9)
- Retrain the ML classifier with more diverse data
- Check camera angle and lighting conditions

### Alert spam
- Increase `alert_cooldown_seconds` (default 300 = 5 minutes)
- Fine-tune confidence threshold

### SMS not received
- Check phone number format (+1XXXXXXXXXX)
- Verify Twilio account has sufficient balance
- Check Twilio logs in Console for delivery status

## Development

### Testing Fall Detection

Use the test script to verify your setup:

```bash
python test_fall_alerts.py
```

### Adding New Alert Types

The alert system is modular. You can extend it by:

1. Adding new classification types in the ML model
2. Creating specific alert handlers in `FallDetectionAlerts`
3. Updating the trigger logic in `classify_poses`

### Custom Alert Channels

Beyond SMS, you can add:
- Email alerts (using SMTP)
- Push notifications (using FCM)
- Webhook calls (for integration with other systems)
- Voice calls (using Twilio Voice API)

## Files

- `src/fall_detection_alerts.py` - Main alert service
- `src/models/yolov8n_pose.py` - Vision service with fall detection integration
- `test_fall_alerts.py` - Test script for verifying setup
- `viam_config_fall_alerts.json` - Example configuration
- `pose_classifier.joblib` - Trained ML model for pose classification

## Security Notes

### 🔒 Credential Security
- **NEVER** commit Twilio credentials to version control
- **USE** environment variables for production deployments  
- **SECURE** environment files with proper permissions (600)
- **ROTATE** credentials regularly
- **MONITOR** Twilio usage in console for unexpected activity

### 🔐 Environment Variable Setup
```bash
# Option 1: User profile (recommended for development)
echo 'export TWILIO_ACCOUNT_SID="ACxxxxx..."' >> ~/.bashrc
echo 'export TWILIO_AUTH_TOKEN="your_token"' >> ~/.bashrc
echo 'export TWILIO_FROM_PHONE="+15551234567"' >> ~/.bashrc  
echo 'export TWILIO_TO_PHONES="+15557654321,+15559876543"' >> ~/.bashrc
source ~/.bashrc

# Option 2: System-wide secure file (recommended for production)
sudo mkdir -p /etc/viam
sudo tee /etc/viam/secrets.env << EOF
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM_PHONE=+15551234567
TWILIO_TO_PHONES=+15557654321,+15559876543
EOF
sudo chmod 600 /etc/viam/secrets.env
sudo chown root:root /etc/viam/secrets.env

# Then modify viam-agent service to load the file:
sudo systemctl edit viam-agent
# Add: [Service]
#      EnvironmentFile=/etc/viam/secrets.env
```

### 🛡️ Additional Security Considerations
- Consider rate limiting for alert endpoints
- Images are stored locally in temp directory - implement cleanup
- Consider encryption for sensitive data transmission
- Use Twilio's IP allowlisting in production
- Set up Twilio webhook signatures for enhanced security
