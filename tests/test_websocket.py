import socketio
import time
import sys

# Create SocketIO Client
sio = socketio.Client()

@sio.event(namespace='/arduino')
def connect():
    print("✅ Connected to Server as Arduino")
    # Send initial status
    sio.emit('telemetry', {
        'battery_voltage': 12.5,
        'status': 'IDLE',
        'last_action': 'BOOT'
    }, namespace='/arduino')

@sio.event(namespace='/arduino')
def connect_error(data):
    print(f"❌ Connection Error: {data}")

@sio.event(namespace='/arduino')
def disconnect():
    print("❌ Disconnected from Server")

@sio.on('command', namespace='/arduino')
def on_command(data):
    print(f"📩 Received Command: {data}")
    
    # Simulate processing
    if data.get('cmd') == 'pick':
        print("   Simulating Pick...")
        time.sleep(1)
        # Send result
        sio.emit('pick_result', {
            'id': data.get('id'),
            'status': 'SUCCESS',
            'result': data.get('class'),
            'to_bin': 'right' if data.get('class') == 'ripe' else 'left'
        }, namespace='/arduino')

def main():
    try:
        print("Connecting to localhost:5000...")
        sio.connect('http://localhost:5000', namespaces=['/arduino'])
        
        # Keep alive
        while True:
            time.sleep(2)
            sio.emit('telemetry', {
                'battery_voltage': 12.4,
                'status': 'IDLE',
                'last_action': 'HEARTBEAT'
            }, namespace='/arduino')
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
