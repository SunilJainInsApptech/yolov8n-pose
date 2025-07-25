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
        
        # Check vision services and their dependencies
        vision_services = [service for service in config.get('services', []) 
                          if service.get('model') == 'rig-guardian:yolov8n-pose:yolov8n-pose']
        
        if not vision_services:
            print("❌ No vision services found in configuration")
            return False
        
        print(f"✅ Found {len(vision_services)} vision service(s)")
        
        # Check camera components
        camera_components = [comp for comp in config.get('components', []) if comp.get('type') == 'camera']
        camera_names = [comp.get('name') for comp in camera_components]
        
        print(f"✅ Camera components found: {camera_names}")
        
        # Check each vision service
        config_issues = []
        for i, service in enumerate(vision_services):
            service_name = service.get('name', f'service_{i}')
            depends_on = service.get('depends_on', [])
            
            print(f"\n  🔍 Vision Service: {service_name}")
            print(f"    Depends on: {depends_on}")
            
            if not depends_on:
                config_issues.append(f"Service '{service_name}' missing 'depends_on' section")
                continue
            
            if len(depends_on) > 1:
                config_issues.append(f"Service '{service_name}' depends on multiple cameras: {depends_on}")
                print(f"    ⚠️  WARNING: Multiple camera dependencies can cause wrong camera names in alerts")
                print(f"    🔧 RECOMMENDATION: Use separate vision services for each camera")
            
            # Check if dependencies match actual cameras
            for dep in depends_on:
                if dep not in camera_names:
                    config_issues.append(f"Service '{service_name}' dependency '{dep}' not found in camera components")
                else:
                    print(f"    ✅ Dependency '{dep}' matches camera component")
        
        if config_issues:
            print(f"\n❌ Configuration Issues Found:")
            for issue in config_issues:
                print(f"  - {issue}")
            return False
        else:
            print(f"\n✅ All configuration checks passed!")
            return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_camera_name_logic():
    """Test the camera name extraction logic."""
    print("\n🔍 Testing Camera Name Logic...")
    
    # Test cases based on current configuration
    test_cases = [
        {
            "name": "Single vision service with single camera dependency",
            "dependencies": ["Lobby_Center_North"],
            "extra": {"camera_name": "Lobby_Center_North"},
            "expected": "Lobby_Center_North",
            "scenario": "Normal case - should work correctly"
        },
        {
            "name": "Single vision service with multiple camera dependencies (PROBLEM CASE)",
            "dependencies": ["Lobby_Center_North", "CPW_Awning_N_Facing"],
            "extra": {},
            "expected": "Lobby_Center_North",  # Will pick first camera
            "scenario": "This causes wrong camera names in alerts!"
        },
        {
            "name": "Separate vision services (RECOMMENDED SOLUTION)",
            "dependencies": ["Lobby_Center_North"],  # Each service has only one camera
            "extra": {"camera_name": "Lobby_Center_North"},
            "expected": "Lobby_Center_North",
            "scenario": "Correct setup - one vision service per camera"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Test: {test_case['name']}")
        print(f"    Scenario: {test_case['scenario']}")
        
        dependencies = test_case['dependencies']
        extra = test_case['extra']
        
        # Simulate the logic from the vision service
        camera_name = "unknown_camera"
        
        if extra and "camera_name" in extra:
            camera_name = extra["camera_name"]
            print(f"    ✅ Used camera_name from extra: {camera_name}")
        else:
            if dependencies:
                if len(dependencies) == 1:
                    camera_name = dependencies[0]
                    print(f"    ✅ Used single camera from dependencies: {camera_name}")
                else:
                    camera_name = dependencies[0]  # Pick first
                    print(f"    ⚠️  Multiple cameras, picked first: {camera_name}")
                    print(f"    ⚠️  Available cameras: {dependencies}")
                    print(f"    ⚠️  This could be wrong camera!")
        
        print(f"    Result: SMS would show 'Camera: {camera_name}'")
        
        if camera_name == test_case['expected']:
            print(f"    ✅ Expected result achieved")
        else:
            print(f"    ❌ Unexpected result (expected: {test_case['expected']})")

def recommend_solution():
    """Provide recommendations for fixing camera identification."""
    print("\n" + "=" * 60)
    print("📋 RECOMMENDED SOLUTION FOR CORRECT CAMERA IDENTIFICATION")
    print("=" * 60)
    
    print("\n🎯 PROBLEM:")
    print("  When one vision service depends on multiple cameras, it cannot")
    print("  determine which camera actually detected the fall, so it always")
    print("  reports the first camera in the dependencies list.")
    
    print("\n✅ SOLUTION:")
    print("  Use separate vision services for each camera:")
    print("  ")
    print("  1. Each vision service should depend on only ONE camera")
    print("  2. Each camera gets its own dedicated vision service")
    print("  3. Camera name will always be correct in fall alerts")
    
    print("\n� CURRENT CONFIGURATION (Updated):")
    print("  - lobby-center-north-vision → depends_on: ['Lobby_Center_North']")
    print("  - cpw-awning-vision → depends_on: ['CPW_Awning_N_Facing']")
    
    print("\n📝 NEXT STEPS:")
    print("  1. Apply the updated configuration")
    print("  2. Restart viam-agent: sudo systemctl restart viam-agent")
    print("  3. Check logs for camera dependency messages")
    print("  4. Test fall detection on each camera separately")
    print("  5. Verify SMS shows correct camera name")

def main():
    """Run all tests."""
    print("🚨 Fall Detection Camera Name Test Suite")
    print("=" * 60)
    
    config_ok = test_camera_config()
    test_camera_name_logic()
    recommend_solution()
    
    print("\n" + "=" * 60)
    if config_ok:
        print("✅ Configuration tests passed!")
    else:
        print("❌ Configuration tests failed!")
        print("   Please fix the configuration issues above")

if __name__ == "__main__":
    main()
