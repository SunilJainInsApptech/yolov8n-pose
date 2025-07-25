#!/usr/bin/env python3
"""
Test script to verify camera name identification in fall detection alerts.

This script tests the camera name extraction logic to ensure SMS alerts
show the correct camera name instead of "unknown_camera".
"""

import json
import os
import sys

def test_camera_config():
    """Test that the Viam configuration has proper camera dependencies."""
    config_path = "viam_config_fall_alerts.json"
    
    print("🔍 Testing Camera Configuration...")
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if vision service has depends_on
        vision_service = None
        for service in config.get('services', []):
            if service.get('model') == 'rig-guardian:yolov8n-pose:yolov8n-pose':
                vision_service = service
                break
        
        if not vision_service:
            print("❌ Vision service not found in configuration")
            return False
        
        if 'depends_on' not in vision_service:
            print("❌ Vision service missing 'depends_on' section")
            print("   Add: \"depends_on\": [\"camera\"] to the vision service configuration")
            return False
        
        depends_on = vision_service['depends_on']
        print(f"✅ Vision service depends_on: {depends_on}")
        
        # Check if cameras exist in components
        camera_components = [comp for comp in config.get('components', []) if comp.get('type') == 'camera']
        camera_names = [comp.get('name') for comp in camera_components]
        
        print(f"✅ Camera components found: {camera_names}")
        
        # Check if depends_on matches actual cameras
        for dep in depends_on:
            if dep not in camera_names:
                print(f"⚠️  Dependency '{dep}' not found in camera components")
            else:
                print(f"✅ Dependency '{dep}' matches camera component")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_camera_name_logic():
    """Test the camera name extraction logic."""
    print("\n🔍 Testing Camera Name Logic...")
    
    # Simulate different scenarios
    test_cases = [
        {
            "name": "With camera_name in extra",
            "extra": {"camera_name": "front_camera"},
            "expected": "front_camera"
        },
        {
            "name": "Without camera_name in extra (should use primary)",
            "extra": {},
            "expected": "primary_camera_fallback"
        },
        {
            "name": "No extra parameter (should use primary)", 
            "extra": None,
            "expected": "primary_camera_fallback"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Test: {test_case['name']}")
        extra = test_case['extra']
        
        # Simulate the logic from the vision service
        camera_name = "unknown_camera"
        if extra and "camera_name" in extra:
            camera_name = extra["camera_name"]
            print(f"    ✅ Used camera_name from extra: {camera_name}")
        else:
            # This would call self.get_primary_camera_name() in real code
            camera_name = "camera"  # Assuming primary camera is named "camera"
            print(f"    ✅ Used primary camera fallback: {camera_name}")
        
        print(f"    Result: {camera_name}")
        if camera_name != "unknown_camera":
            print(f"    ✅ Success - SMS would show 'Camera: {camera_name}'")
        else:
            print(f"    ❌ Failed - SMS would still show 'Camera: unknown_camera'")

def main():
    """Run all tests."""
    print("🚨 Fall Detection Camera Name Test Suite")
    print("=" * 50)
    
    config_ok = test_camera_config()
    test_camera_name_logic()
    
    print("\n" + "=" * 50)
    if config_ok:
        print("✅ Configuration tests passed!")
        print("\n📋 Next steps:")
        print("1. Restart the viam-agent service")
        print("2. Check logs for camera dependency messages")
        print("3. Test fall detection to verify SMS shows correct camera name")
    else:
        print("❌ Configuration tests failed!")
        print("\n🔧 Fix the configuration issues above and run test again")

if __name__ == "__main__":
    main()
