#!/usr/bin/env python3
"""
Web Interface Launcher with Custom Port
======================================

Launches the web interface on a specified port to avoid conflicts.
"""

import os
import sys
import argparse
from web_interface import app

def main():
    parser = argparse.ArgumentParser(description="Start Web Interface on Custom Port")
    parser.add_argument("--port", type=int, default=5001, help="Port to run on (default: 5001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    print("🌐 Starting Web Interface")
    print("=" * 50)
    print(f"🌐 Web Interface: http://localhost:{args.port}")
    print(f"📁 Upload folder: datasets/")
    print(f"💾 Models folder: models/")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(debug=False, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
