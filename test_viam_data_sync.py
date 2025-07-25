#!/usr/bin/env python3
"""
Test script to verify Viam data sync is working correctly.
"""

import os
import json
import time
from datetime import datetime

def test_capture_directory():
    """Test if the capture directory exists and is writable."""
    capture_dir = "/home/sunil/Documents/viam_captured_images"
    
    print("🔍 Testing Viam Data Capture Setup...")
    print(f"Target directory: {capture_dir}")
    
    # Check if directory exists
    if os.path.exists(capture_dir):
        print(f"✅ Directory exists")
        
        # Check if writable
        if os.access(capture_dir, os.W_OK):
            print(f"✅ Directory is writable")
        else:
            print(f"❌ Directory is not writable")
            return False
    else:
        print(f"❌ Directory does not exist")
        try:
            os.makedirs(capture_dir, exist_ok=True)
            print(f"✅ Created directory: {capture_dir}")
        except Exception as e:
            print(f"❌ Failed to create directory: {e}")
            return False
    
    # List current files
    try:
        files = os.listdir(capture_dir)
        print(f"📂 Current files in directory: {len(files)}")
        
        # Show fall detection images
        fall_files = [f for f in files if f.startswith('fall_')]
        if fall_files:
            print(f"🚨 Fall detection images found: {len(fall_files)}")
            for f in fall_files[-5:]:  # Last 5
                filepath = os.path.join(capture_dir, f)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  - {f} ({size} bytes, {mtime_str})")
        else:
            print(f"📷 No fall detection images found yet")
            
    except Exception as e:
        print(f"❌ Could not list directory: {e}")
        return False
    
    return True

def test_data_manager_config():
    """Check if data manager is properly configured."""
    print(f"\n🔍 Checking Data Manager Configuration...")
    
    # This would typically be in your robot config
    print("Expected data manager config:")
    print('  "name": "data_manager-1"')
    print('  "capture_dir": "/home/sunil/Documents/viam_captured_images"')
    print('  "sync_interval_mins": 1')
    print('  "tags": ["Fall"]')
    
    # Check viam agent status
    print(f"\n🔍 To check viam-agent status, run:")
    print(f"  sudo systemctl status viam-agent")
    print(f"  sudo journalctl -u viam-agent -f")

def create_test_image():
    """Create a test image to verify sync."""
    capture_dir = "/home/sunil/Documents/viam_captured_images"
    
    print(f"\n🧪 Creating test image...")
    
    # Create a small test file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_sync_{timestamp}.txt"
    filepath = os.path.join(capture_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            f.write(f"Test file created at {datetime.now()}\n")
            f.write("This file tests Viam data sync.\n")
        
        print(f"✅ Test file created: {filepath}")
        print(f"🔄 Check Viam app Data tab in 1-2 minutes for this file")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test file: {e}")
        return False

def main():
    """Run all tests."""
    print("🚨 Viam Data Sync Test Suite")
    print("=" * 50)
    
    # Test 1: Directory setup
    dir_ok = test_capture_directory()
    
    # Test 2: Config info
    test_data_manager_config()
    
    # Test 3: Create test file
    if dir_ok:
        create_test_image()
    
    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Check viam-agent logs: sudo journalctl -u viam-agent -f")
    print("2. Wait 1-2 minutes for sync")
    print("3. Check Viam app → Data tab → Look for 'Fall' tag")
    print("4. Trigger a test fall to see if images appear")

if __name__ == "__main__":
    main()
