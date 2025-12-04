import requests
import time
import subprocess
import sys
import os

def test_endpoints():
    base_url = "http://localhost:5000"
    
    print("Testing /api/camera/list...")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/api/camera/list")
        duration = time.time() - start_time
        if response.status_code == 200:
            print(f"✅ /api/camera/list returned 200 in {duration:.2f}s")
            print(response.json())
        else:
            print(f"❌ /api/camera/list failed with {response.status_code}")
            print(response.text)
            
        # Test cache
        print("\nTesting cached /api/camera/list...")
        start_time = time.time()
        response = requests.get(f"{base_url}/api/camera/list")
        duration = time.time() - start_time
        print(f"✅ Cached /api/camera/list returned in {duration:.2f}s")
        
    except Exception as e:
        print(f"❌ /api/camera/list failed: {e}")

    print("\nTesting /api/camera/switch...")
    try:
        # Try switching to index 0
        response = requests.post(f"{base_url}/api/camera/switch", json={"index": 0})
        if response.status_code == 200:
            print("✅ /api/camera/switch (index 0) returned 200")
            print(response.json())
        else:
            print(f"❌ /api/camera/switch failed with {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ /api/camera/switch failed: {e}")

    print("\nTesting /api/camera/feed...")
    try:
        response = requests.head(f"{base_url}/api/camera/feed")
        if response.status_code == 200:
            print("✅ /api/camera/feed returned 200")
        else:
            print(f"❌ /api/camera/feed failed with {response.status_code}")
    except Exception as e:
        print(f"❌ /api/camera/feed failed: {e}")

if __name__ == "__main__":
    # Start the server in the background
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "web_interface.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        test_endpoints()
    finally:
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait()
