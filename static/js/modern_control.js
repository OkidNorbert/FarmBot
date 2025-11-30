// WebSocket connection
const socket = io();
let isConnected = false;
let currentSpeed = 100;

// Servo configuration
const servoConfig = {
    base: { min: 0, max: 180, default: 90 },
    forearm: { min: 10, max: 170, default: 90 },
    shoulder: { min: 15, max: 165, default: 90 },
    elbow: { min: 15, max: 165, default: 90 },
    pitch: { min: 20, max: 160, default: 90 },
    claw: { min: 0, max: 90, default: 0 },
    speed: { min: 33, max: 100, default: 100 }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeSliders();
    setupEventListeners();
    initializeCamera();
});

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('✅ Socket.IO connected');
    updateConnectionStatus(true);
});

socket.on('disconnect', () => {
    console.log('❌ Socket.IO disconnected');
    updateConnectionStatus(false);
});

socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error);
    updateConnectionStatus(false);
});

socket.on('status', (data) => {
    console.log('Status update:', data);
    if (data.connected !== undefined) {
        updateConnectionStatus(data.connected);
    }
    if (data.message) {
        showNotification(data.message, data.connected ? 'success' : 'info');
    }
});

socket.on('telemetry', (data) => {
    updateTelemetry(data);
});

socket.on('error', (data) => {
    console.error('Socket error:', data);
    showNotification(data.message || 'An error occurred', 'danger');
});

// Initialize sliders
function initializeSliders() {
    document.querySelectorAll('.servo-slider').forEach(slider => {
        const servo = slider.dataset.servo;
        const value = slider.value;
        updateSliderDisplay(servo, value);

        // Initialize tracking variables for dynamic speed
        slider.dataset.lastValue = value;
        slider.dataset.lastTime = Date.now();

        // Add highlight listeners
        slider.addEventListener('mousedown', () => highlightServo(servo));
        slider.addEventListener('touchstart', () => highlightServo(servo));
        slider.addEventListener('mouseup', () => unhighlightServo(servo));
        slider.addEventListener('touchend', () => unhighlightServo(servo));
        slider.addEventListener('mouseleave', () => unhighlightServo(servo));

        slider.addEventListener('input', function () {
            const value = parseInt(this.value);
            updateSliderDisplay(servo, value);

            if (servo === 'speed') {
                currentSpeed = value;
                // Send speed command immediately
                sendServoCommand(servo, value);
                return;
            }

            // Calculate dynamic speed based on slider movement
            const now = Date.now();
            const lastTime = parseInt(this.dataset.lastTime || now);
            const lastValue = parseInt(this.dataset.lastValue || value);
            const deltaTime = now - lastTime;

            let dynamicSpeed = currentSpeed; // Default to global speed

            // If moved quickly (within 200ms), calculate speed
            if (deltaTime > 0 && deltaTime < 200) {
                const deltaValue = Math.abs(value - lastValue);
                const rawSpeed = (deltaValue / deltaTime) * 200;
                dynamicSpeed = Math.min(Math.max(parseInt(rawSpeed), 10), 120);
            }

            // Update tracking variables
            this.dataset.lastValue = value;
            this.dataset.lastTime = now;

            // Debounce servo commands
            clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => {
                // Only send command if connected
                if (socket.connected && isConnected) {
                    sendServoCommand(servo, value, dynamicSpeed);
                } else {
                    console.warn(`Cannot send command for ${servo}: not connected`);
                }
            }, 50);
        });
    });
}

function highlightServo(servo) {
    const group = document.getElementById(`${servo}-group`);
    if (group) {
        group.classList.add('servo-highlight');
    }
}

function unhighlightServo(servo) {
    const group = document.getElementById(`${servo}-group`);
    if (group) {
        group.classList.remove('servo-highlight');
    }
}

// Update slider display
function updateSliderDisplay(servo, value) {
    const valueElement = document.getElementById(`${servo}Value`);
    if (valueElement) {
        if (servo === 'speed') {
            valueElement.textContent = `${value}%`;
        } else {
            valueElement.textContent = `${value}°`;
            updateArmVisualization(servo, value);
        }
    }
}

// Update Arm Visualization (2D side view) - Real-time responsive updates
function updateArmVisualization(servo, angle) {
    const group = document.getElementById(`${servo}-group`);
    if (!group) {
        console.warn(`Visualization group not found for servo: ${servo}`);
        return;
    }

    let rotation = 0;
    let centerX = 200;
    let centerY = 450;

    switch (servo) {
        case 'base':
            // Base rotation is around Z axis - rotate entire arm assembly
            const baseRect = group.querySelector('rect');
            if (baseRect) {
                // Visual feedback: change color intensity based on angle
                const intensity = Math.abs(angle - 90) / 90; // 0 to 1
                const hue = 210 + (intensity * 30); // Blue to cyan
                baseRect.setAttribute('fill', `hsl(${hue}, 70%, ${50 + intensity * 20}%)`);
            }
            // Rotate entire arm assembly for base rotation
            const armAssembly = document.getElementById('arm-assembly');
            if (armAssembly) {
                rotation = angle - 90; // Convert 0-180° to -90 to +90°
                armAssembly.setAttribute('transform', `rotate(${rotation} 200 450)`);
            }
            // Add visual feedback
            group.classList.add('servo-moving');
            setTimeout(() => group.classList.remove('servo-moving'), 200);
            return;
            
        case 'shoulder':
            centerY = 450; // Base of shoulder (top of base)
            rotation = angle - 90; // Convert 0-180° to -90 to +90°
            break;
            
        case 'forearm':
            centerY = 350; // Joint between shoulder and forearm
            rotation = angle - 90;
            break;
            
        case 'elbow':
            centerY = 250; // Elbow joint
            rotation = angle - 90;
            break;
            
        case 'pitch':
            centerY = 150; // Wrist pitch joint
            rotation = angle - 90;
            break;
            
        case 'claw':
            // Claw opens/closes
            // angle: 0 = closed, 90 = open
            const openAmount = angle / 90; // 0 to 1
            const leftFinger = group.querySelector('.claw-finger-left');
            const rightFinger = group.querySelector('.claw-finger-right');

            if (leftFinger && rightFinger) {
                // Move fingers apart
                // Left moves left (-X), Right moves right (+X)
                const offset = openAmount * 15; // Increased for better visibility
                leftFinger.setAttribute('transform', `translate(-${offset}, 0)`);
                rightFinger.setAttribute('transform', `translate(${offset}, 0)`);
                
                // Also change color intensity
                const clawCircle = group.querySelector('circle');
                if (clawCircle) {
                    const brightness = 30 + (openAmount * 30);
                    clawCircle.setAttribute('fill', `hsl(0, 70%, ${brightness}%)`);
                }
            }
            // Add visual feedback
            group.classList.add('servo-moving');
            setTimeout(() => group.classList.remove('servo-moving'), 200);
            return;
    }

    // Apply rotation with smooth transition
    group.setAttribute('transform', `rotate(${rotation} ${centerX} ${centerY})`);
    
    // Add visual feedback - highlight the moving part
    group.classList.add('servo-moving');
    setTimeout(() => {
        group.classList.remove('servo-moving');
    }, 200);
}

// Send servo command via WebSocket
function sendServoCommand(servo, angle, speed) {
    // Check if socket is connected
    if (!socket.connected) {
        console.warn('Socket not connected, cannot send command');
        showNotification('Not connected to server. Please click Connect.', 'warning');
        return;
    }
    
    // Map frontend servo names to backend expected names
    // Backend expects: base, forearm, arm, wrist_yaw, wrist_pitch, claw
    // Frontend uses: base, forearm, shoulder, elbow, pitch, claw
    const servoMap = {
        'shoulder': 'arm',  // Frontend 'shoulder' -> backend 'arm'
        'elbow': 'wrist_yaw',  // Frontend 'elbow' -> backend 'wrist_yaw'
        'pitch': 'wrist_pitch',  // Frontend 'pitch' -> backend 'wrist_pitch'
        // Others map directly
        'base': 'base',
        'forearm': 'forearm',
        'claw': 'claw',
        'speed': 'speed'  // Speed is handled separately
    };
    
    const backendServo = servoMap[servo] || servo;
    
    // Don't send speed as a servo command
    if (servo === 'speed') {
        const command = {
            cmd: 'set_speed',
            speed: angle  // For speed, angle parameter is the speed value
        };
        console.log('📤 Sending speed command:', command);
        socket.emit('servo_command', command);
        return;
    }
    
    const command = {
        cmd: 'move',
        servo: backendServo,
        angle: parseInt(angle),
        speed: speed ? parseInt(speed) : currentSpeed
    };
    
    console.log('📤 Sending servo command:', command);
    socket.emit('servo_command', command);
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('btnConnect').addEventListener('click', connectArduino);
    document.getElementById('btnDisconnect').addEventListener('click', disconnectArduino);
    document.getElementById('btnStart').addEventListener('click', startAutoMode);
    document.getElementById('btnSave').addEventListener('click', savePose);
    document.getElementById('btnReset').addEventListener('click', resetArm);
    document.getElementById('modeToggle').addEventListener('change', toggleMode);
}

// Connect to Arduino
function connectArduino() {
    if (!socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        return;
    }
    socket.emit('servo_command', { cmd: 'connect' });
    showNotification('Connecting to Arduino...', 'info');
}

// Disconnect from Arduino
function disconnectArduino() {
    socket.emit('servo_command', { cmd: 'disconnect' });
    updateConnectionStatus(false);
}

// Start automatic mode
function startAutoMode() {
    socket.emit('servo_command', { cmd: 'start' });
    showNotification('Starting automatic mode...', 'success');
}

// Save current pose
function savePose() {
    const pose = {};
    document.querySelectorAll('.servo-slider').forEach(slider => {
        const servo = slider.dataset.servo;
        if (servo !== 'speed') {
            pose[servo] = parseInt(slider.value);
        }
    });

    socket.emit('servo_command', {
        cmd: 'save',
        pose: pose,
        name: `Pose_${new Date().getTime()}`
    });
    showNotification('Pose saved successfully!', 'success');
}

// Reset arm to home position
function resetArm() {
    if (confirm('Reset all servos to home position?')) {
        socket.emit('servo_command', { cmd: 'reset' });

        // Reset sliders to default positions
        Object.keys(servoConfig).forEach(servo => {
            if (servo !== 'speed') {
                const slider = document.getElementById(`${servo}Slider`);
                const defaultValue = servoConfig[servo].default;
                if (slider) {
                    slider.value = defaultValue;
                    updateSliderDisplay(servo, defaultValue);
                }
            }
        });

        showNotification('Arm reset to home position', 'info');
    }
}

// Toggle mode
function toggleMode(event) {
    const isAuto = event.target.checked;
    socket.emit('servo_command', {
        cmd: 'set_mode',
        mode: isAuto ? 'auto' : 'manual'
    });
    showNotification(`Switched to ${isAuto ? 'Automatic' : 'Manual'} mode`, 'info');
}

// Update connection status
function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusBadge = document.getElementById('connectionStatus');
    const statusDot = statusBadge.querySelector('.status-dot');
    const statusText = statusBadge.querySelector('span:last-child');

    if (connected) {
        statusBadge.className = 'connection-badge connected';
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Connected';

        // Enable buttons
        document.getElementById('btnConnect').disabled = true;
        document.getElementById('btnDisconnect').disabled = false;
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnSave').disabled = false;
        document.getElementById('btnReset').disabled = false;
    } else {
        statusBadge.className = 'connection-badge disconnected';
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Disconnected';

        // Disable buttons
        document.getElementById('btnConnect').disabled = false;
        document.getElementById('btnDisconnect').disabled = true;
        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnSave').disabled = true;
        document.getElementById('btnReset').disabled = true;
    }
}

// Update telemetry
function updateTelemetry(data) {
    if (data.distance_mm !== undefined) {
        document.getElementById('distanceValue').textContent = `${data.distance_mm} mm`;
    }

    if (data.status) {
        document.getElementById('statusValue').textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    }

    if (data.arduino_connected !== undefined) {
        updateConnectionStatus(data.arduino_connected);
    }

    if (data.mode) {
        const modeToggle = document.getElementById('modeToggle');
        modeToggle.checked = (data.mode === 'auto');
    }

    // Update last update time
    const now = new Date();
    document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
}

// Initialize camera feed
function initializeCamera() {
    const cameraFeed = document.getElementById('cameraFeed');
    const placeholder = document.getElementById('cameraPlaceholder');

    cameraFeed.onload = function () {
        placeholder.style.display = 'none';
        cameraFeed.style.display = 'block';
    };

    cameraFeed.onerror = function () {
        placeholder.style.display = 'flex';
        cameraFeed.style.display = 'none';
    };
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.style.minWidth = '300px';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alertDiv);

    setTimeout(() => {
        alertDiv.remove();
    }, 3000);
}
