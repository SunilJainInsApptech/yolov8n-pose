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
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send fall detection alert via Twilio SMS"""
        
        try:
            # Check if we should send alert
            if not self.should_send_alert(person_id, confidence):
                return False
            
            # Record alert time
            timestamp = datetime.now()
            self.last_alert_time[person_id] = timestamp
            
            # Save image locally
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
