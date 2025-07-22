#!/usr/bin/env python3
"""
Test script to verify Viam environment variable configuration for Twilio.
This script simulates how the YOLOv8 module loads environment variables.
"""

import os
import json
from pathlib import Path

def test_environment_variables():
    """Test if Twilio environment variables are properly configured."""
    print("🧪 Testing Viam Environment Variable Configuration for Twilio")
    print("=" * 60)
    
    # Check if environment variables are available
    env_vars = {
        'TWILIO_ACCOUNT_SID': os.environ.get('TWILIO_ACCOUNT_SID'),
        'TWILIO_AUTH_TOKEN': os.environ.get('TWILIO_AUTH_TOKEN'),
        'TWILIO_FROM_PHONE': os.environ.get('TWILIO_FROM_PHONE'),
        'TWILIO_TO_PHONES': os.environ.get('TWILIO_TO_PHONES'),
        'TWILIO_WEBHOOK_URL': os.environ.get('TWILIO_WEBHOOK_URL')
    }
    
    print("Environment Variable Status:")
    print("-" * 30)
    
    required_vars = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_PHONE', 'TWILIO_TO_PHONES']
    all_required_set = True
    
    for var_name, value in env_vars.items():
        is_required = var_name in required_vars
        status = "✅ SET" if value else "❌ NOT_SET"
        required_text = " (REQUIRED)" if is_required else " (optional)"
        
        if is_required and not value:
            all_required_set = False
            
        print(f"{var_name:20} : {status}{required_text}")
        
        # Show partial values for verification (without exposing full credentials)
        if value and var_name == 'TWILIO_ACCOUNT_SID':
            print(f"{'':20}   Value: {value[:8]}... (length: {len(value)})")
        elif value and var_name == 'TWILIO_AUTH_TOKEN':
            print(f"{'':20}   Value: {value[:4]}... (length: {len(value)})")
        elif value and var_name in ['TWILIO_FROM_PHONE', 'TWILIO_TO_PHONES', 'TWILIO_WEBHOOK_URL']:
            print(f"{'':20}   Value: {value}")
    
    print("\n" + "=" * 60)
    
    if all_required_set:
        print("✅ SUCCESS: All required Twilio environment variables are configured!")
        print("   Fall detection alerts should work properly.")
        
        # Test parsing TWILIO_TO_PHONES
        to_phones = env_vars['TWILIO_TO_PHONES'].split(',') if env_vars['TWILIO_TO_PHONES'] else []
        print(f"   📱 Alert will be sent to {len(to_phones)} phone number(s):")
        for i, phone in enumerate(to_phones, 1):
            print(f"      {i}. {phone.strip()}")
            
    else:
        print("❌ FAILURE: Missing required environment variables!")
        print("   Fall detection alerts will be DISABLED.")
        print("\n📋 Next Steps:")
        print("   1. Check your Viam robot configuration file")
        print("   2. Ensure agent.advanced_settings.viam_server_env is configured")
        print("   3. Restart viam-agent after making changes")
        print("   4. See VIAM_ENVIRONMENT_VARIABLES.md for detailed setup")

def test_configuration_files():
    """Check if configuration files are properly set up."""
    print("\n🔧 Testing Configuration Files")
    print("=" * 60)
    
    config_files = [
        'viam_config_fall_alerts.json',
        'viam_config_fall_alerts_template.json'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Found: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                # Check if it has the new agent.advanced_settings structure
                if 'agent' in config and 'advanced_settings' in config['agent']:
                    if 'viam_server_env' in config['agent']['advanced_settings']:
                        print(f"   🔒 Uses secure environment variables")
                        env_config = config['agent']['advanced_settings']['viam_server_env']
                        twilio_vars = [k for k in env_config.keys() if k.startswith('TWILIO_')]
                        print(f"   📱 Configured Twilio variables: {len(twilio_vars)}")
                    else:
                        print(f"   ⚠️  Missing viam_server_env section")
                else:
                    print(f"   ⚠️  No agent.advanced_settings section (old format)")
                    
                # Check service configuration
                if 'services' in config:
                    for service in config['services']:
                        if service.get('model') == 'rig-guardian:yolov8n-pose:yolov8n-pose':
                            attrs = service.get('attributes', {})
                            use_env = attrs.get('use_env_for_twilio', False)
                            print(f"   🎛️  use_env_for_twilio: {use_env}")
                            break
                            
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing error: {e}")
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"❌ Missing: {config_file}")

if __name__ == "__main__":
    test_environment_variables()
    test_configuration_files()
    
    print("\n" + "=" * 60)
    print("🏁 Test Complete!")
    print("\nFor detailed setup instructions, see:")
    print("   📖 VIAM_ENVIRONMENT_VARIABLES.md") 
    print("   📖 FALL_DETECTION_README.md")
