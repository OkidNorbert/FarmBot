// WebSocket connection
let socket = null;
let isConnected = false;
let currentSpeed = 100;
let pickMode = false;
let pickModeCooldown = false;


// Initialize SocketIO connection with error handling
function initializeSocketIO() {
    try {
        // Check if Socket.IO library is loaded
        if (typeof io === 'undefined') {
            console.error('Socket.IO library not loaded. Please check if the CDN script is loaded.');
            if (typeof showNotification === 'function') {
                showNotification('Socket.IO library not found. Please refresh the page.', 'danger');
            }
            if (typeof updateConnectionStatus === 'function') {
                updateConnectionStatus(false, 'Socket.IO library not loaded');
            }
            return false;
        }

        console.log('Initializing Socket.IO connection...');

        // Initialize socket connection
        socket = io({
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 10,
            timeout: 20000,
            forceNew: false,
            autoConnect: true
        });

        setupSocketHandlers();

        // Set initial connection status
        if (typeof updateConnectionStatus === 'function') {
            updateConnectionStatus(false, 'Connecting...');
        }

        return true;
    } catch (error) {
        console.error('Failed to initialize SocketIO:', error);
        if (typeof showNotification === 'function') {
            showNotification('Failed to connect to server: ' + error.message, 'danger');
        }
        if (typeof updateConnectionStatus === 'function') {
            updateConnectionStatus(false, 'Connection failed');
        }
        return false;
    }
}

// Setup Socket.IO event handlers
function setupSocketHandlers() {
    if (!socket) return;

    socket.on('connect', () => {
        console.log('✅ Socket.IO connected');
        isConnected = true;
        updateConnectionStatus(true, 'Socket.IO Connected');
    });

    socket.on('disconnect', (reason) => {
        console.log('❌ Socket.IO disconnected:', reason);
        isConnected = false;
        updateConnectionStatus(false);
    });

    socket.on('connect_error', (error) => {
        console.error('Socket.IO connection error:', error);
        isConnected = false;
        updateConnectionStatus(false);
        showNotification('Connection error. Retrying...', 'warning');
    });

    socket.on('reconnect', (attemptNumber) => {
        console.log('✅ Socket.IO reconnected after', attemptNumber, 'attempts');
        isConnected = true;
        updateConnectionStatus(true);
        showNotification('Reconnected to server', 'success');
    });

    socket.on('reconnect_error', (error) => {
        console.error('Socket.IO reconnection error:', error);
    });

    socket.on('reconnect_failed', () => {
        console.error('Socket.IO reconnection failed');
        showNotification('Failed to reconnect. Please refresh the page.', 'danger');
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
        console.log('📡 Received telemetry event:', data);
        updateTelemetry(data);
    });

    socket.on('error', (data) => {
        console.error('Socket error:', data);
        if (data.message) {
            showNotification('Error: ' + data.message, 'danger');
        }
    });
}

// Servo configuration
const servoConfig = {
    'waist': { min: 0, max: 180, default: 90, unit: '°' },
    'shoulder': { min: 15, max: 165, default: 90, unit: '°' },
    'elbow': { min: 10, max: 180, default: 90, unit: '°' },
    'wrist_roll': { min: 15, max: 165, default: 90, unit: '°' },
    'wrist_pitch': { min: 0, max: 160, default: 90, unit: '°' },
    'claw': { min: 10, max: 115, default: 115, unit: '°' },
    'speed': { min: 1, max: 180, default: 20, unit: ' deg/s' }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    try {
        // Initialize SocketIO first - wait a bit for SocketIO library to load
        setTimeout(() => {
            if (!initializeSocketIO()) {
                console.warn('SocketIO initialization failed, but continuing with page setup');
                updateConnectionStatus(false, 'SocketIO init failed');
            } else {
                // Wait for socket connection before enabling connect button
                const checkConnection = setInterval(() => {
                    if (socket && socket.connected) {
                        clearInterval(checkConnection);
                        console.log('SocketIO connected, ready for commands');
                    } else if (socket && socket.disconnected) {
                        // Socket initialized but not connected yet, keep waiting
                        console.log('Waiting for SocketIO connection...');
                    }
                }, 100);

                // Stop checking after 10 seconds
                setTimeout(() => clearInterval(checkConnection), 10000);
            }
        }, 100);

        if (typeof initializeSliders === 'function') {
            initializeSliders();
        }
        if (typeof setupEventListeners === 'function') {
            setupEventListeners();
        }
        if (typeof initializeCamera === 'function') {
            initializeCamera();
        }

        // Initialize arm orientation display after sliders are initialized
        setTimeout(() => {
            if (typeof updateArmOrientation === 'function') {
                updateArmOrientation();
            }
        }, 200);
    } catch (error) {
        console.error('Error initializing control page:', error);
    }
});

// Update telemetry display function is defined later in the file

// Initialize sliders
function initializeSliders() {
    try {
        const sliders = document.querySelectorAll('.servo-slider');
        if (sliders.length === 0) {
            console.warn('No servo sliders found on page');
            return;
        }
        sliders.forEach(slider => {
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

                // Update arm orientation when shoulder or forearm changes
                if (servo === 'shoulder' || servo === 'elbow') { // Updated for new joint names
                    updateArmOrientation();
                }

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
    } catch (error) {
        console.error('Error initializing sliders:', error);
    }
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

// Determine arm orientation (front/back) based on shoulder and elbow angles
function determineArmOrientation() {
    const shoulderSlider = document.getElementById('shoulderSlider');
    const elbowSlider = document.getElementById('elbowSlider');

    if (!shoulderSlider || !elbowSlider) {
        return null;
    }

    const shoulderAngle = parseInt(shoulderSlider.value);
    const elbowAngle = parseInt(elbowSlider.value);

    // Front: shoulder and elbow are 90-180 degrees
    // Back: shoulder and elbow are 90-0 degrees
    const isShoulderFront = shoulderAngle >= 90 && shoulderAngle <= 180;
    const isElbowFront = elbowAngle >= 90 && elbowAngle <= 180;
    const isShoulderBack = shoulderAngle >= 0 && shoulderAngle <= 90;
    const isElbowBack = elbowAngle >= 0 && elbowAngle <= 90;

    // Determine orientation based on both servos
    if (isShoulderFront && isElbowFront) {
        return 'front';
    } else if (isShoulderBack && isElbowBack) {
        return 'back';
    } else {
        // Mixed orientation - determine based on which is more dominant
        const shoulderFrontness = (shoulderAngle - 90) / 90; // 0 to 1 for front
        const elbowFrontness = (elbowAngle - 90) / 90; // 0 to 1 for front
        const avgFrontness = (shoulderFrontness + elbowFrontness) / 2;

        return avgFrontness > 0 ? 'front' : 'back';
    }
}

// Update arm orientation display
function updateArmOrientation() {
    const orientation = determineArmOrientation();
    const orientationElement = document.getElementById('armOrientation');

    if (orientationElement && orientation) {
        orientationElement.textContent = orientation === 'front' ? 'Front' : 'Back';
        orientationElement.className = orientation === 'front'
            ? 'badge bg-success'
            : 'badge bg-info';
    }
}

// Update slider display
function updateSliderDisplay(servo, value) {
    const valueElement = document.getElementById(`${servo}Value`);
    if (valueElement) {
        const config = servoConfig[servo];
        if (config) {
            valueElement.textContent = `${value}${config.unit}`;
        } else {
            valueElement.textContent = `${value}`; // Fallback if unit not defined
        }
        updateArmVisualization(servo, value);

        // Update arm orientation when shoulder or elbow changes
        if (servo === 'shoulder' || servo === 'elbow') {
            updateArmOrientation();
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
    let centerY = 450; // Default for waist

    switch (servo) {
        case 'waist':
            // Waist rotation is around Z axis - rotate entire arm assembly
            const waistRect = group.querySelector('rect');
            if (waistRect) {
                // Visual feedback: change color intensity based on angle
                const intensity = Math.abs(angle - 90) / 90; // 0 to 1
                const hue = 210 + (intensity * 30); // Blue to cyan
                waistRect.setAttribute('fill', `hsl(${hue}, 70%, ${50 + intensity * 20}%)`);
            }
            // Rotate entire arm assembly for waist rotation
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
            centerY = 450; // Base of shoulder (top of waist)
            rotation = angle - 90; // Convert 0-180° to -90 to +90°
            break;

        case 'elbow':
            centerY = 350; // Joint between shoulder and elbow
            rotation = angle - 90;
            break;

        case 'wrist_roll':
            centerY = 250; // Joint between elbow and wrist_roll
            rotation = angle - 90;
            break;

        case 'wrist_pitch':
            centerY = 150; // Joint between wrist_roll and wrist_pitch
            rotation = angle - 90;
            break;

        case 'claw':
            // Claw opens/closes
            // angle: 60 = closed, 150 = open (based on new config)
            const openAmount = (angle - servoConfig.claw.min) / (servoConfig.claw.max - servoConfig.claw.min); // 0 to 1
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
    if (!socket || !socket.connected) {
        console.warn('Cannot send command: Socket.IO not connected');
        return;
    }
    // Check if socket is connected
    if (!socket.connected) {
        console.warn('Socket not connected, cannot send command');
        showNotification('Not connected to server. Please click Connect.', 'warning');
        return;
    }

    // Map frontend servo names to Arduino expected names
    const servoMap = {
        'waist': 'waist',
        'shoulder': 'shoulder',
        'elbow': 'elbow',
        'wrist_roll': 'wrist_roll',
        'wrist_pitch': 'wrist_pitch',
        'claw': 'claw'
    };

    const backendServo = servoMap[servo] || servo;

    // Don't send speed as a servo command
    if (servo === 'speed') {
        const command = {
            cmd: 'set_speed',
            speed: angle  // For speed, angle parameter is the speed value
        };
        console.log('📤 Sending speed command:', command);
        if (socket && socket.connected) {
            socket.emit('servo_command', command);
        } else {
            console.warn('Cannot send command: Socket.IO not connected');
        }
        return;
    }

    const command = {
        cmd: 'move',
        servo: backendServo,
        angle: parseInt(angle),
        speed: speed ? parseInt(speed) : currentSpeed
    };

    // Track movement if recording
    trackServoChange(servo, parseInt(angle));

    console.log('📤 Sending servo command:', command);
    if (socket && socket.connected) {
        socket.emit('servo_command', command);
    } else {
        console.warn('Cannot send command: Socket.IO not connected');
    }
}

// Setup event listeners
function setupEventListeners() {
    try {
        const btnConnect = document.getElementById('btnConnect');
        const btnDisconnect = document.getElementById('btnDisconnect');
        const btnStart = document.getElementById('btnStart');
        const btnSave = document.getElementById('btnSave');
        const btnCapturePose = document.getElementById('btnCapturePose');
        const btnReset = document.getElementById('btnReset');

        if (btnConnect) btnConnect.addEventListener('click', connectArduino);
        if (btnDisconnect) btnDisconnect.addEventListener('click', disconnectArduino);
        if (btnStart) btnStart.addEventListener('click', startRecordingMovements);
        if (btnSave) btnSave.addEventListener('click', savePose);
        if (btnCapturePose) btnCapturePose.addEventListener('click', capturePose);
        if (btnReset) btnReset.addEventListener('click', resetArm);

        const modeToggle = document.getElementById('modeToggle');
        if (modeToggle) modeToggle.addEventListener('change', toggleAutomaticMode);
    } catch (error) {
        console.error('Error setting up event listeners:', error);
    }
}

// Connect to Arduino
function connectArduino() {
    // Check if socket is initialized
    if (!socket) {
        showNotification('Socket.IO not initialized. Initializing...', 'warning');
        if (initializeSocketIO()) {
            // Wait a bit for connection
            setTimeout(() => {
                if (socket && socket.connected) {
                    socket.emit('servo_command', { cmd: 'connect' });
                    showNotification('Connecting to Arduino...', 'info');
                } else {
                    showNotification('Socket.IO connection failed. Please refresh the page.', 'danger');
                }
            }, 1000);
        } else {
            showNotification('Failed to initialize Socket.IO. Please refresh the page.', 'danger');
        }
        return;
    }

    // Check if socket is connected
    if (!socket.connected) {
        showNotification('Socket.IO not connected. Attempting to connect...', 'warning');
        socket.connect();
        // Wait for connection and retry
        const retryConnect = () => {
            if (socket.connected) {
                socket.emit('servo_command', { cmd: 'connect' });
                showNotification('Connecting to Arduino...', 'info');
            } else {
                setTimeout(() => {
                    if (socket.connected) {
                        socket.emit('servo_command', { cmd: 'connect' });
                        showNotification('Connecting to Arduino...', 'info');
                    } else {
                        showNotification('Socket.IO connection timeout. Please check server status.', 'danger');
                    }
                }, 2000);
            }
        };
        socket.once('connect', retryConnect);
        return;
    }

    // Socket is connected, proceed with connect command
    socket.emit('servo_command', { cmd: 'connect' });
    showNotification('Connecting to Arduino...', 'info');
}

// Disconnect from Arduino
function disconnectArduino() {
    if (!socket) {
        showNotification('Socket.IO not initialized', 'warning');
        return;
    }
    socket.emit('servo_command', { cmd: 'disconnect' });
    updateConnectionStatus(false);
}

// Record arm movements (Start/Stop recording)
let isRecording = false;
let recordedMovements = [];
let recordingStartTime = null;

function startRecordingMovements() {
    if (!socket || !socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        return;
    }

    if (isRecording) {
        // Stop recording
        isRecording = false;
        const btnStart = document.getElementById('btnStart');
        if (btnStart) {
            btnStart.innerHTML = '<i class="fas fa-play"></i> Start Recording';
            btnStart.classList.remove('btn-danger');
            btnStart.classList.add('btn-primary');
        }

        // Save recorded movements
        if (recordedMovements.length > 0) {
            socket.emit('servo_command', {
                cmd: 'save_recording',
                movements: recordedMovements,
                duration: Date.now() - recordingStartTime,
                name: `Recording_${new Date().toISOString().replace(/[:.]/g, '-')}`
            });
            showNotification(`Recording stopped. Saved ${recordedMovements.length} movements.`, 'success');
        } else {
            showNotification('Recording stopped (no movements recorded).', 'info');
        }

        recordedMovements = [];
        recordingStartTime = null;
        // Notify backend that recording stopped so any temporary servo overrides are reverted
        try {
            if (socket && socket.connected) {
                socket.emit('servo_command', { cmd: 'record_stop' });
            }
        } catch (e) {
            console.warn('Failed to notify backend about recording stop:', e);
        }
    } else {
        // Start recording
        isRecording = true;
        recordedMovements = [];
        recordingStartTime = Date.now();

        const btnStart = document.getElementById('btnStart');
        if (btnStart) {
            btnStart.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
            btnStart.classList.remove('btn-primary');
            btnStart.classList.add('btn-danger');
        }

        showNotification('Recording arm movements... Move the arm to record.', 'info');
        // Notify backend that recording started so servos can be enabled if needed
        try {
            if (socket && socket.connected) {
                socket.emit('servo_command', { cmd: 'record_start' });
            }
        } catch (e) {
            console.warn('Failed to notify backend about recording start:', e);
        }
    }
}

// Track servo changes during recording
function trackServoChange(servo, angle) {
    if (isRecording && recordingStartTime) {
        const timestamp = Date.now() - recordingStartTime;
        recordedMovements.push({
            timestamp: timestamp,
            servo: servo,
            angle: angle
        });
    }
}

// Save current pose
function savePose() {
    if (!socket || !socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        return;
    }

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

// Capture pose from latest telemetry (prefer backend-reported angles)
function capturePose() {
    if (!socket || !socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        return;
    }

    // Prefer telemetry-provided servo angles if available
    const telemetryAngles = window.latestServoAngles || null;
    const pose = {};

    if (telemetryAngles && Object.keys(telemetryAngles).length > 0) {
        // Copy known keys (ensure consistent servo keys)
        ['waist', 'shoulder', 'elbow', 'wrist_roll', 'wrist_pitch', 'claw'].forEach(k => {
            if (telemetryAngles[k] !== undefined && telemetryAngles[k] !== null) {
                pose[k] = parseInt(telemetryAngles[k]);
            }
        });
    }

    // Fill missing values from sliders
    document.querySelectorAll('.servo-slider').forEach(slider => {
        const servo = slider.dataset.servo;
        if (servo !== 'speed' && pose[servo] === undefined) {
            pose[servo] = parseInt(slider.value);
        }
    });

    // Optional prompt for a friendly name
    const name = prompt('Enter a name for the captured pose (optional):', `Pose_${new Date().getTime()}`) || `Pose_${new Date().getTime()}`;

    socket.emit('servo_command', {
        cmd: 'save',
        pose: pose,
        name: name
    });

    showNotification('Pose capture requested — saved on server.', 'success');
}

// Reset arm to home position
function resetArm() {
    if (!socket || !socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        return;
    }

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

// Toggle automatic mode (for tomato detection and picking)
function toggleAutomaticMode(event) {
    const isAuto = event.target.checked;

    if (!socket || !socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        event.target.checked = !isAuto; // Revert toggle
        return;
    }

    // Send automatic mode command to backend
    fetch('/api/auto/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    }).then(response => response.json())
        .then(data => {
            if (data.success) {
                if (isAuto) {
                    showNotification('Automatic mode started: Detecting and picking ready tomatoes', 'success');
                } else {
                    // Stop automatic mode
                    fetch('/api/auto/stop', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    }).then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                showNotification('Automatic mode stopped', 'info');
                            }
                        });
                }
            } else {
                showNotification('Failed to toggle automatic mode: ' + (data.error || 'Unknown error'), 'danger');
                event.target.checked = !isAuto; // Revert toggle
            }
        }).catch(error => {
            console.error('Error toggling automatic mode:', error);
            showNotification('Error toggling automatic mode: ' + error.message, 'danger');
            event.target.checked = !isAuto; // Revert toggle
        });
}

// Legacy toggle mode (kept for compatibility)
function toggleMode(event) {
    if (!socket || !socket.connected) {
        showNotification('Socket.IO not connected. Please wait...', 'warning');
        // Revert toggle
        event.target.checked = !event.target.checked;
        return;
    }

    const isAuto = event.target.checked;
    socket.emit('servo_command', {
        cmd: 'set_mode',
        mode: isAuto ? 'auto' : 'manual'
    });
    showNotification(`Switched to ${isAuto ? 'Automatic' : 'Manual'} mode`, 'info');
}

// Update connection status
function updateConnectionStatus(connected, message = null) {
    isConnected = connected;
    const statusBadge = document.getElementById('connectionStatus');
    if (!statusBadge) {
        console.warn('Connection status badge not found');
        return;
    }

    const statusDot = statusBadge.querySelector('.status-dot');
    const statusText = statusBadge.querySelector('span:last-child');

    if (connected) {
        statusBadge.className = 'connection-badge connected';
        if (statusDot) statusDot.className = 'status-dot connected';
        if (statusText) statusText.textContent = message || 'Connected';

        // Enable buttons
        const btnConnect = document.getElementById('btnConnect');
        const btnDisconnect = document.getElementById('btnDisconnect');
        const btnStart = document.getElementById('btnStart');
        const btnSave = document.getElementById('btnSave');
        const btnReset = document.getElementById('btnReset');

        if (btnConnect) btnConnect.disabled = true;
        if (btnDisconnect) btnDisconnect.disabled = false;
        if (btnStart) btnStart.disabled = false;
        if (btnSave) btnSave.disabled = false;
        if (btnCapturePose) btnCapturePose.disabled = false;
        if (btnReset) btnReset.disabled = false;
    } else {
        statusBadge.className = 'connection-badge disconnected';
        if (statusDot) statusDot.className = 'status-dot disconnected';
        if (statusText) statusText.textContent = message || 'Disconnected';

        // Disable buttons
        const btnConnect = document.getElementById('btnConnect');
        const btnDisconnect = document.getElementById('btnDisconnect');
        const btnStart = document.getElementById('btnStart');
        const btnSave = document.getElementById('btnSave');
        const btnReset = document.getElementById('btnReset');

        if (btnConnect) btnConnect.disabled = false;
        if (btnDisconnect) btnDisconnect.disabled = true;
        if (btnStart) btnStart.disabled = true;
        if (btnSave) btnSave.disabled = true;
        if (btnCapturePose) btnCapturePose.disabled = true;
        if (btnReset) btnReset.disabled = true;
    }
}

// Update telemetry
function updateTelemetry(data) {
    console.log('📊 Telemetry update received:', data);
    // Store latest servo angles for capture usage
    try {
        if (data && data.servo_angles) {
            window.latestServoAngles = data.servo_angles;
        } else {
            // Keep previous telemetry if new packet doesn't include servo_angles
            window.latestServoAngles = window.latestServoAngles || {};
        }
    } catch (e) {
        window.latestServoAngles = window.latestServoAngles || {};
    }

    // Update distance (ToF)
    const distanceElement = document.getElementById('distanceValue');
    if (distanceElement) {
        if (data.distance_mm !== undefined && data.distance_mm !== null && data.distance_mm > 0) {
            distanceElement.textContent = `${data.distance_mm} mm`;
        } else {
            distanceElement.textContent = '-- mm';
        }
    } else {
        console.warn('distanceValue element not found');
    }

    // Update status
    if (data.status) {
        const statusElement = document.getElementById('statusValue');
        if (statusElement) {
            const status = data.status.toLowerCase();
            statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);

            // Add visual styling based on status
            if (status === 'running') {
                statusElement.innerHTML = `<span class="badge bg-success pulse-animation">Running</span>`;
            } else if (status === 'idle') {
                statusElement.innerHTML = `<span class="badge bg-secondary">Idle</span>`;
            } else if (status === 'disconnected') {
                statusElement.innerHTML = `<span class="badge bg-danger">Disconnected</span>`;
            }
        } else {
            console.warn('statusValue element not found');
        }
    }

    // Update connection status
    if (data.arduino_connected !== undefined) {
        updateConnectionStatus(data.arduino_connected);
    }

    // Update mode toggle
    if (data.mode) {
        const modeToggle = document.getElementById('modeToggle');
        if (modeToggle) {
            modeToggle.checked = (data.mode === 'auto');
        }
    }

    // Update arm orientation from backend
    if (data.arm_orientation) {
        const orientationElement = document.getElementById('armOrientation');
        if (orientationElement) {
            const orientation = data.arm_orientation === 'front' ? 'Front' : 'Back';
            orientationElement.innerHTML = `<span class="badge ${data.arm_orientation === 'front' ? 'bg-success' : 'bg-info'}">${orientation}</span>`;
        }
    }

    // Always update last update time when telemetry is received
    const lastUpdateElement = document.getElementById('lastUpdate');
    if (lastUpdateElement) {
        const now = new Date();
        lastUpdateElement.textContent = now.toLocaleTimeString();
    } else {
        console.warn('lastUpdate element not found');
    }
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

// Toggle Pick Mode
function togglePickMode() {
    const toggle = document.getElementById('pickModeToggle');
    pickMode = toggle.checked;

    const cameraContainer = document.getElementById('cameraContainer');
    if (pickMode) {
        cameraContainer.style.cursor = 'crosshair';
        cameraContainer.title = 'Click to Pick';
        showNotification('Pick Mode Enabled: Click on the camera feed to pick an item.', 'info');
    } else {
        cameraContainer.style.cursor = 'default';
        cameraContainer.title = '';
        showNotification('Pick Mode Disabled.', 'secondary');
    }
}

// Initialize Pick Mode click handler
function initializePickMode() {
    const cameraFeed = document.getElementById('cameraFeed');
    if (!cameraFeed) return;

    cameraFeed.addEventListener('click', async function (e) {
        if (!pickMode || pickModeCooldown) return;

        // Cooldown to prevent multiple accidental clicks
        pickModeCooldown = true;
        setTimeout(() => { pickModeCooldown = false; }, 2000);

        const rect = cameraFeed.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Convert to image coordinates
        const imgWidth = cameraFeed.naturalWidth || 640;
        const imgHeight = cameraFeed.naturalHeight || 480;
        const scaleX = imgWidth / rect.width;
        const scaleY = imgHeight / rect.height;

        const pixelX = Math.round(x * scaleX);
        const pixelY = Math.round(y * scaleY);

        console.log(`Pick Mode click: (${pixelX}, ${pixelY})`);
        showNotification(`Picking at (${pixelX}, ${pixelY})...`, 'info');

        try {
            const response = await fetch('/api/control/pick_at_pixel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: pixelX, y: pixelY })
            });

            const result = await response.json();
            if (result.success) {
                showNotification(`Success: ${result.message}`, 'success');
            } else {
                showNotification(`Pick failed: ${result.message}`, 'danger');
            }
        } catch (error) {
            console.error('Error in click-to-pick:', error);
            showNotification('Error sending pick command', 'danger');
        }
    });
}

// Global initialization
document.addEventListener('DOMContentLoaded', () => {
    initializeSocketIO();
    initializeCamera();
    initializePickMode();
});
