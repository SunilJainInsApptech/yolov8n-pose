"""
Fall Detection Alert Service using Twilio
Sends SMS alerts when a fall is detected with image and metadata
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import base64
import tempfile
from twilio.rest import Client
from viam.media.video import ViamImage

# Try to import Viam DataManager service
try:
    from viam.services.data_manager import DataManager
    VIAM_DATA_AVAILABLE = True
except ImportError:
    VIAM_DATA_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

class FallDetectionAlerts:
    """Service for sending fall detection alerts via Twilio"""
    
    def __init__(self, config: dict):
        """Initialize Twilio client and alert configuration"""
        # Try to load from environment variables first, then config
        self.account_sid = (
            os.environ.get('TWILIO_ACCOUNT_SID') or 
            config.get('twilio_account_sid')
        )
        self.auth_token = (
            os.environ.get('TWILIO_AUTH_TOKEN') or 
            config.get('twilio_auth_token')
        )
        self.from_phone = (
            os.environ.get('TWILIO_FROM_PHONE') or 
            config.get('twilio_from_phone')
        )
        
        # Handle phone numbers from environment (comma-separated) or config (list)
        env_phones = os.environ.get('TWILIO_TO_PHONES')
        if env_phones:
            self.to_phones = [phone.strip() for phone in env_phones.split(',')]
        else:
            self.to_phones = config.get('twilio_to_phones', [])
        
        self.webhook_url = (
            os.environ.get('TWILIO_WEBHOOK_URL') or 
            config.get('webhook_url')
        )
        
        # Alert settings (these can stay in config since they're not sensitive)
        self.min_confidence = config.get('fall_confidence_threshold', 0.7)
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 300)
        self.last_alert_time = {}  # Track last alert time per person
        
        # Push notification settings
        self.notify_service_sid = (
            os.environ.get('TWILIO_NOTIFY_SERVICE_SID') or
            config.get('twilio_notify_service_sid')
        )
        self.push_notification_url = (
            os.environ.get('RIGGUARDIAN_WEBHOOK_URL') or
            config.get('rigguardian_webhook_url', 'https://rigguardian.com/api/fall-alert')
        )
        
        # Log what source we're using (without exposing credentials)
        if os.environ.get('TWILIO_ACCOUNT_SID'):
            LOGGER.info("✅ Using Twilio credentials from environment variables")
        else:
            LOGGER.info("⚠️ Using Twilio credentials from robot configuration")
        
        # Validate required config
        if not all([self.account_sid, self.auth_token, self.from_phone]):
            raise ValueError("Missing required Twilio configuration: account_sid, auth_token, from_phone")
        
        if not self.to_phones:
            raise ValueError("No alert phone numbers configured")
        
        # Initialize Twilio client
        try:
            self.client = Client(self.account_sid, self.auth_token)
            LOGGER.info("✅ Twilio client initialized successfully")
        except Exception as e:
            LOGGER.error(f"❌ Failed to initialize Twilio client: {e}")
            raise
    
    def should_send_alert(self, person_id: str, confidence: float) -> bool:
        """Check if we should send an alert based on confidence and cooldown"""
        # Check confidence threshold
        if confidence < self.min_confidence:
            LOGGER.debug(f"Fall confidence {confidence:.3f} below threshold {self.min_confidence}")
            return False
        
        # Check cooldown period
        now = datetime.now()
        if person_id in self.last_alert_time:
            time_since_last = (now - self.last_alert_time[person_id]).total_seconds()
            if time_since_last < self.cooldown_seconds:
                LOGGER.debug(f"Alert cooldown active for person {person_id} ({time_since_last:.1f}s < {self.cooldown_seconds}s)")
                return False
        
        return True
    
    async def save_image_locally(self, image: ViamImage, person_id: str) -> str:
        """Save image to local temporary file and return path"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_detection_{person_id}_{timestamp}.jpg"
            
            # Save to temporary directory
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, filename)
            
            # Convert ViamImage to bytes and save
            with open(image_path, 'wb') as f:
                f.write(image.data)
            
            LOGGER.info(f"📸 Fall detection image saved: {image_path}")
            return image_path
            
        except Exception as e:
            LOGGER.error(f"❌ Failed to save image: {e}")
            return ""
    
    def format_alert_message(self, 
                           camera_name: str, 
                           person_id: str, 
                           confidence: float,
                           timestamp: datetime,
                           image_path: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format the alert message for SMS"""
        
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🚨 FALL DETECTED 🚨\n"
        message += f"Camera: {camera_name}\n"
        # message += f"Person: {person_id}\n"
        message += f"Fall Confidence: {confidence:.1%}\n"
        message += f"Time: {timestamp_str}\n"
        
        if metadata:
            if 'probabilities' in metadata:
                probs = metadata['probabilities']
                # message += f"Pose Probs: "
                # message += f"Fall:{probs.get('fallen', 0):.1%}\n"
        #        message += f"Stand:{probs.get('standing', 0):.1%} "
        #       message += f"Sit:{probs.get('sitting', 0):.1%}\n"
        
        if image_path:
            message += f"Image: {os.path.basename(image_path)}\n"
        message += "\nPlease check the location immediately."
        
        return message
    
    async def send_fall_alert(self, 
                            camera_name: str,
                            person_id: str, 
                            confidence: float,
                            image: ViamImage,
                            metadata: Optional[Dict[str, Any]] = None,
                            data_manager=None) -> bool:
        """Send fall detection alert via Twilio SMS"""
        
        try:
            # Check if we should send alert
            if not self.should_send_alert(person_id, confidence):
                return False
            
            # Record alert time
            timestamp = datetime.now()
            self.last_alert_time[person_id] = timestamp
            
            # Save image to Viam-monitored directory for automatic sync
            await self.save_fall_image(camera_name, person_id, confidence, image, data_manager)
            
            # Save image locally for SMS reference
            image_path = await self.save_image_locally(image, person_id)
            
            # Format alert message
            message = self.format_alert_message(
                camera_name=camera_name,
                person_id=person_id,
                confidence=confidence,
                timestamp=timestamp,
                image_path=image_path,
                metadata=metadata
            )
            
            # Send SMS to all configured phone numbers
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    # Send SMS
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Fall alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send SMS to {phone_number}: {e}")
            
            # Send push notification to rigguardian.com app
            push_success = await self.send_push_notification(
                camera_name=camera_name,
                person_id=person_id,
                confidence=confidence,
                timestamp=timestamp,
                metadata=metadata
            )
            
            if push_success:
                LOGGER.info("📱 Push notification sent to rigguardian.com successfully")
            else:
                LOGGER.warning("⚠️ Push notification failed - SMS alert still sent")
            
            if success_count > 0:
                LOGGER.info(f"✅ Fall alert sent successfully to {success_count}/{len(self.to_phones)} recipients")
                return True
            else:
                LOGGER.error("❌ Failed to send fall alert to any recipients")
                return False
                
        except Exception as e:
            LOGGER.error(f"❌ Error sending fall alert: {e}")
            return False
    
    async def send_test_alert(self, camera_name: str = "test_camera") -> bool:
        """Send a test alert to verify Twilio configuration"""
        try:
            message = f"🧪 TEST ALERT 🧪\n"
            message += f"Fall detection system is active\n"
            message += f"Camera: {camera_name}\n"
            message += f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
            message += f"System is monitoring for falls."
            
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Test alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send test SMS to {phone_number}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            LOGGER.error(f"❌ Error sending test alert: {e}")
            return False
    
    async def send_push_notification(self, camera_name: str, person_id: str, confidence: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send push notification to rigguardian.com web app"""
        try:
            # For web-based apps, use webhook approach (not Twilio Notify)
            # Twilio Notify is for mobile apps only
            return await self.send_webhook_notification(camera_name, person_id, confidence, timestamp, metadata)
                
        except Exception as e:
            LOGGER.error(f"❌ Error sending push notification: {e}")
            return False
    
    async def send_webhook_notification(self, camera_name: str, person_id: str, confidence: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send notification via webhook to rigguardian.com web app"""
        try:
            import aiohttp
            import json
            
            # Create webhook payload optimized for web app notifications
            webhook_data = {
                "alert_type": "fall_detection",
                "timestamp": timestamp.isoformat(),
                "camera_name": camera_name,
                "person_id": person_id,
                "confidence": confidence,
                "severity": "critical",
                "location": camera_name,
                "title": "🚨 Fall Alert - Immediate Action Required",
                "message": f"Fall detected on {camera_name} with {confidence:.1%} confidence",
                "requires_immediate_attention": True,
                "notification_type": "web_push",  # Indicate this is for web app
                "metadata": metadata or {},
                "actions": [
                    {"action": "view_camera", "title": "View Camera"},
                    {"action": "acknowledge", "title": "Acknowledge"},
                    {"action": "dispatch_help", "title": "Send Help"}
                ]
            }
            
            LOGGER.info(f"🔄 Sending webhook notification to rigguardian.com web app")
            LOGGER.info(f"📊 Payload: {camera_name} fall alert at {timestamp.strftime('%H:%M:%S')}")
            
            # Send HTTP POST to rigguardian.com
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.push_notification_url,
                    json=webhook_data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'FallDetectionSystem/1.0',
                        'X-Alert-Type': 'fall_detection',
                        'X-Severity': 'critical'
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        LOGGER.info(f"✅ Web notification sent to rigguardian.com successfully")
                        LOGGER.info(f"📱 Response: {response_text}")
                        return True
                    else:
                        LOGGER.error(f"❌ Webhook failed with status {response.status}")
                        LOGGER.error(f"❌ Response: {response_text}")
                        return False
                        
        except ImportError:
            LOGGER.error("❌ aiohttp not installed - cannot send webhook notifications")
            LOGGER.error("💡 Install with: pip install aiohttp")
            return False
        except aiohttp.ClientTimeout:
            LOGGER.error("❌ Webhook request timed out (>10 seconds)")
            return False
        except aiohttp.ClientError as e:
            LOGGER.error(f"❌ HTTP client error: {e}")
            return False
        except Exception as e:
            LOGGER.error(f"❌ Failed to send webhook notification: {e}")
            return False
    
    async def save_fall_image(self, camera_name: str, person_id: str, confidence: float, image: ViamImage, data_manager=None):
        """Save fall detection image with proper naming for DataManager to sync to dataset"""
        try:
            timestamp = datetime.utcnow()
            DATASET_ID = "68851ef0628dd018729e9541"
            
            LOGGER.info(f"🔄 Saving fall image for DataManager to sync to dataset")
            LOGGER.info(f"� Image size: {len(image.data)} bytes, Component: {camera_name}")
            
            # Use exact timestamp format that Viam data manager expects
            # Format: YYYY-MM-DDTHH:MM:SS.fffffffZ (RFC3339 with microseconds)
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            
            # Viam expects: [timestamp]_[component_name]_[method_name].[extension]
            filename = f"{timestamp_str}_{camera_name}_ReadImage.jpg"
            filepath = f"/home/sunil/Documents/viam_captured_images/{filename}"
            
            LOGGER.info(f"🔄 Saving with Viam naming convention: {filename}")
            LOGGER.info(f"📊 Image size: {len(image.data)} bytes, Component: {camera_name}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image.data)
            
            # Create a .txt metadata file with the same timestamp and component
            metadata_filename = f"{timestamp_str}_{camera_name}_FallData.txt"
            metadata_filepath = f"/home/sunil/Documents/viam_captured_images/{metadata_filename}"
            
            metadata_content = f"""FALL_DETECTION_EVENT
timestamp: {timestamp.isoformat()}
component: {camera_name}
person_id: {person_id}
confidence: {confidence:.3f}
dataset_id: {DATASET_ID}
event_type: fall_detected
method_name: ReadImage
component_type: camera
"""
            
            with open(metadata_filepath, 'w') as meta_f:
                meta_f.write(metadata_content)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                LOGGER.info(f"✅ Fall image saved: {filename} ({file_size} bytes)")
                LOGGER.info(f"� Metadata saved: {metadata_filename}")
                LOGGER.info(f"🎯 Component: {camera_name} → Dataset: 68851ef0628dd018729e9541")
                LOGGER.info("🔄 Files will sync to Viam within 1 minute")
            else:
                LOGGER.error(f"❌ Failed to save: {filepath}")
                
        except Exception as e:
            LOGGER.error(f"❌ Error saving fall image: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
