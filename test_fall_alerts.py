#!/usr/bin/env python3
"""
Test script for fall detection alerts
Tests the Twilio integration without needing actual fall detection
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fall_detection_alerts import FallDetectionAlerts
from viam.media.video import ViamImage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_config():
    """Create test configuration - you'll need to update with your Twilio credentials"""
    return {
        'twilio_account_sid': 'YOUR_TWILIO_ACCOUNT_SID',  # Replace with your Account SID
        'twilio_auth_token': 'YOUR_TWILIO_AUTH_TOKEN',    # Replace with your Auth Token  
        'twilio_from_phone': '+1234567890',               # Replace with your Twilio phone number
        'twilio_to_phones': ['+1987654321'],              # Replace with recipient phone number(s)
        'fall_confidence_threshold': 0.7,
        'alert_cooldown_seconds': 10,  # Shorter for testing
        'webhook_url': 'https://example.com/images'
    }

async def create_test_image():
    """Create a simple test image"""
    # Create a small test image (just dummy data)
    test_image_data = b'\\x89PNG\\r\\n\\x1a\\n' + b'\\x00' * 100  # Minimal PNG-like data
    return ViamImage(test_image_data, "image/png")

async def test_fall_alerts():
    """Test the fall detection alert system"""
    
    logger.info("🧪 Testing Fall Detection Alerts...")
    
    # Create configuration
    config = create_test_config()
    
    # Check if configuration has been updated
    if config['twilio_account_sid'] == 'YOUR_TWILIO_ACCOUNT_SID':
        logger.error("❌ Please update the test configuration with your actual Twilio credentials!")
        logger.error("Edit this script and replace:")
        logger.error("  - YOUR_TWILIO_ACCOUNT_SID with your Account SID")
        logger.error("  - YOUR_TWILIO_AUTH_TOKEN with your Auth Token")
        logger.error("  - Phone numbers with real numbers")
        return False
    
    try:
        # Initialize fall alerts
        logger.info("Initializing FallDetectionAlerts...")
        alerts = FallDetectionAlerts(config)
        
        # Test connection with a test alert
        logger.info("Sending test alert...")
        test_success = await alerts.send_test_alert("test_camera")
        
        if test_success:
            logger.info("✅ Test alert sent successfully!")
            
            # Test fall alert (simulated)
            logger.info("Testing fall alert simulation...")
            test_image = await create_test_image()
            
            fall_success = await alerts.send_fall_alert(
                camera_name="test_camera",
                person_id="test_person_1", 
                confidence=0.85,
                image=test_image,
                metadata={
                    "probabilities": {
                        "fallen": 0.85,
                        "standing": 0.10,
                        "sitting": 0.03,
                        "crouching": 0.02
                    },
                    "features": {"test_feature": 1.0}
                }
            )
            
            if fall_success:
                logger.info("✅ Fall alert sent successfully!")
                return True
            else:
                logger.error("❌ Fall alert failed")
                return False
        else:
            logger.error("❌ Test alert failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def print_setup_instructions():
    """Print setup instructions for Twilio"""
    print("""
🔧 TWILIO SETUP INSTRUCTIONS:

1. Sign up for Twilio account at https://www.twilio.com/
2. Get your Account SID and Auth Token from the Console
3. Purchase a Twilio phone number
4. Update this test script with your credentials
5. Update your robot configuration with the same credentials

📋 Configuration needed:
- twilio_account_sid: Your Account SID from Twilio Console
- twilio_auth_token: Your Auth Token from Twilio Console  
- twilio_from_phone: Your Twilio phone number (e.g., +15551234567)
- twilio_to_phones: List of phone numbers to receive alerts

💡 For testing, you can use Twilio's free trial which gives you credit for SMS messages.

⚙️  Add these to your robot configuration in the vision service attributes:
{
  "twilio_account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "twilio_auth_token": "your_auth_token_here",
  "twilio_from_phone": "+15551234567",
  "twilio_to_phones": ["+15557654321"],
  "fall_confidence_threshold": 0.7,
  "alert_cooldown_seconds": 300
}
""")

if __name__ == "__main__":
    print("🚨 Fall Detection Alert Test Script 🚨")
    print_setup_instructions()
    
    # Run the test
    success = asyncio.run(test_fall_alerts())
    
    if success:
        print("\\n✅ All tests passed! Fall detection alerts are working.")
    else:
        print("\\n❌ Tests failed. Check the logs above for details.")
        print("Make sure to update the Twilio credentials in this script.")
