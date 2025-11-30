// Modern Robotic Arm Controller JavaScript

class RoboticArmController {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.mode = 'manual'; // 'manual' or 'auto'
        this.servoValues = {
            base: 90,
            forearm: 90,
            arm: 90,
            wrist_yaw: 90,
            wrist_pitch: 90,
            claw: 0,
            speed: 100  // Default to max speed (60 deg/s) for responsive control
        };
        
        // Track slider movement velocity for dynamic speed control
        this.sliderVelocity = {}; // Track velocity per servo
        this.lastSliderValues = {}; // Track last values
        this.lastSliderTimes = {}; // Track last update times
        
        // Debouncing/throttling for command sending
        this.commandTimeouts = {}; // Per-servo timeout IDs
        this.lastCommandTimes = {}; // Track when last command was sent
        this.commandThrottleMs = 50; // Minimum time between commands (20 commands/sec max)
        this.lastSpeedSent = null; // Track last speed sent to avoid redundant SPEED commands
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateSliderFills();
        this.updateTelemetry();
        this.setupCameraFeed();
    }
    
    setupCameraFeed() {
        // Setup camera feed
        const cameraFeed = document.getElementById('cameraFeed');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        
        if (cameraFeed) {
            cameraFeed.onload = () => {
                if (cameraPlaceholder) {
                    cameraPlaceholder.style.display = 'none';
                }
                if (cameraFeed) {
                    cameraFeed.style.display = 'block';
                }
            };
            
            cameraFeed.onerror = () => {
                if (cameraPlaceholder) {
                    cameraPlaceholder.style.display = 'flex';
                }
                if (cameraFeed) {
                    cameraFeed.style.display = 'none';
                }
            };
        }
    }
    
    setupEventListeners() {
        try {
            // Connection buttons
            const btnConnect = document.getElementById('btnConnect');
            const btnDisconnect = document.getElementById('btnDisconnect');
            const btnStart = document.getElementById('btnStart');
            const btnSave = document.getElementById('btnSave');
            const btnReset = document.getElementById('btnReset');
            
            if (btnConnect) {
                btnConnect.addEventListener('click', () => this.connect());
            } else {
                console.error('btnConnect not found');
            }
            
            if (btnDisconnect) {
                btnDisconnect.addEventListener('click', () => this.disconnect());
            }
            
            if (btnStart) {
                btnStart.addEventListener('click', () => this.start());
            }
            
            if (btnSave) {
                btnSave.addEventListener('click', () => this.save());
            }
            
            if (btnReset) {
                btnReset.addEventListener('click', () => this.reset());
            }
            
            // Mode toggle
            const modeToggle = document.getElementById('modeToggle');
            if (modeToggle) {
                modeToggle.addEventListener('change', (e) => {
                    this.mode = e.target.checked ? 'auto' : 'manual';
                    this.updateMode();
                });
            }
            
            // Servo sliders
            const sliders = document.querySelectorAll('.servo-slider');
            console.log(`Found ${sliders.length} servo sliders`);
            sliders.forEach(slider => {
                slider.addEventListener('input', (e) => this.onSliderChange(e));
                slider.addEventListener('mousedown', () => this.onSliderStart());
                slider.addEventListener('mouseup', () => this.onSliderEnd());
            });
            
            console.log('Event listeners set up successfully');
        } catch (error) {
            console.error('Error setting up event listeners:', error);
        }
    }
    
    connect() {
        if (this.socket && this.socket.connected) {
            console.log('Already connected');
            return;
        }
        
        console.log('Attempting to connect to server...');
        
        // Connect to Flask-SocketIO
        try {
            this.socket = io({
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionAttempts: 5
            });
            
            this.socket.on('connect', () => {
                console.log('✅ Connected to server');
                this.connected = true;
                this.updateConnectionStatus(true);
                // Send connect command after a short delay to ensure connection is stable
                setTimeout(() => {
                    this.sendCommand({ cmd: 'connect' });
                }, 100);
            });
            
            this.socket.on('disconnect', (reason) => {
                console.log('❌ Disconnected from server:', reason);
                this.connected = false;
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('connect_error', (error) => {
                console.error('Connection error:', error);
                this.connected = false;
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('telemetry', (data) => {
                this.handleTelemetry(data);
            });
            
            this.socket.on('status', (data) => {
                console.log('Status update:', data);
                this.handleStatus(data);
            });
            
            this.socket.on('error', (error) => {
                console.error('Socket error:', error);
            });
        } catch (error) {
            console.error('Failed to create socket connection:', error);
            alert('Failed to connect: ' + error.message);
        }
    }
    
    disconnect() {
        if (this.socket) {
            this.sendCommand({ cmd: 'disconnect' });
            this.socket.disconnect();
            this.socket = null;
        }
        this.connected = false;
        this.updateConnectionStatus(false);
    }
    
    sendCommand(command) {
        if (!this.socket) {
            console.warn('Socket not initialized, cannot send command:', command);
            return;
        }
        
        if (!this.socket.connected) {
            console.warn('Not connected, cannot send command:', command);
            return;
        }
        
        console.log('📤 Sending command:', command);
        try {
            this.socket.emit('servo_command', command);
        } catch (error) {
            console.error('Error sending command:', error);
        }
    }
    
    onSliderChange(event) {
        const slider = event.target;
        const servoName = slider.dataset.servo;
        const value = parseInt(slider.value);
        const currentTime = Date.now();
        
        // Update local value
        this.servoValues[servoName] = value;
        
        // Update display
        this.updateServoValue(servoName, value);
        this.updateSliderFill(servoName, slider);
        
        // Calculate movement velocity (degrees per second)
        let velocity = 0;
        if (this.lastSliderValues[servoName] !== undefined && this.lastSliderTimes[servoName] !== undefined) {
            const deltaAngle = Math.abs(value - this.lastSliderValues[servoName]);
            const deltaTime = (currentTime - this.lastSliderTimes[servoName]) / 1000; // Convert to seconds
            if (deltaTime > 0) {
                velocity = deltaAngle / deltaTime; // Degrees per second
            }
        }
        
        // Store current values for next calculation
        this.lastSliderValues[servoName] = value;
        this.lastSliderTimes[servoName] = currentTime;
        this.sliderVelocity[servoName] = velocity;
        
        // Send command if connected
        if (this.connected) {
            if (servoName === 'speed') {
                // Speed is handled separately
                this.sendCommand({
                    cmd: 'set_speed',
                    speed: value
                });
            } else if (this.mode === 'manual') {
                // Throttle command sending to reduce lag
                const now = Date.now();
                const lastSent = this.lastCommandTimes[servoName] || 0;
                const timeSinceLastCommand = now - lastSent;
                
                // Clear any pending timeout for this servo
                if (this.commandTimeouts[servoName]) {
                    clearTimeout(this.commandTimeouts[servoName]);
                }
                
                // Calculate dynamic speed based on velocity
                // Use initialization speed (60 deg/s) as maximum for smooth, responsive control
                const MAX_SPEED = 60; // Match initialization speed
                const MIN_SPEED = 20; // Minimum for smooth movement
                
                const baseSpeed = this.servoValues.speed || 50; // Base speed from speed slider (0-100%)
                const baseSpeedDegPerSec = Math.round((baseSpeed / 100) * MAX_SPEED); // Convert % to deg/s
                
                // Calculate dynamic speed: fast slider = fast servo, slow slider = slow servo
                let dynamicSpeed = MIN_SPEED;
                if (velocity > 30) {
                    // Fast movement - use velocity-based speed (capped at MAX_SPEED)
                    dynamicSpeed = Math.min(Math.round(velocity), MAX_SPEED);
                } else if (velocity > 10) {
                    // Medium movement - blend velocity with base speed
                    dynamicSpeed = Math.max(Math.round(velocity * 0.7 + baseSpeedDegPerSec * 0.3), MIN_SPEED);
                } else {
                    // Slow movement - use base speed
                    dynamicSpeed = Math.max(baseSpeedDegPerSec, MIN_SPEED);
                }
                
                // Throttle: only send if enough time has passed OR if this is the final position
                const sendCommand = () => {
                    this.lastCommandTimes[servoName] = Date.now();
                    
                    // Only send SPEED command if it changed
                    if (this.lastSpeedSent !== dynamicSpeed) {
                        this.sendCommand({
                            cmd: 'set_speed',
                            speed: dynamicSpeed
                        });
                        this.lastSpeedSent = dynamicSpeed;
                    }
                    
                    // Send angle command
                    this.sendCommand({
                        cmd: 'move',
                        servo: servoName,
                        angle: value,
                        speed: dynamicSpeed
                    });
                };
                
                if (timeSinceLastCommand >= this.commandThrottleMs) {
                    // Send immediately if enough time has passed
                    sendCommand();
                } else {
                    // Schedule to send after throttle period
                    this.commandTimeouts[servoName] = setTimeout(sendCommand, this.commandThrottleMs - timeSinceLastCommand);
                }
            }
        }
    }
    
    onSliderStart() {
        // Visual feedback when dragging
        document.body.style.cursor = 'grabbing';
    }
    
    onSliderEnd() {
        document.body.style.cursor = 'default';
    }
    
    updateServoValue(servoName, value) {
        const valueElement = document.getElementById(`${servoName}Value`);
        if (valueElement) {
            if (servoName === 'speed') {
                valueElement.textContent = `${value}%`;
            } else {
                valueElement.textContent = `${value}°`;
            }
        }
    }
    
    updateSliderFill(servoName, slider) {
        const fill = document.getElementById(`${servoName}Fill`);
        if (fill) {
            const min = parseFloat(slider.min);
            const max = parseFloat(slider.max);
            const value = parseFloat(slider.value);
            const percentage = ((value - min) / (max - min)) * 100;
            fill.style.width = `${percentage}%`;
        }
    }
    
    updateSliderFills() {
        const sliders = document.querySelectorAll('.servo-slider');
        sliders.forEach(slider => {
            const servoName = slider.dataset.servo;
            this.updateSliderFill(servoName, slider);
        });
    }
    
    start() {
        if (!this.connected) {
            this.showToast('Please connect first', 'error');
            return;
        }
        
        if (this.mode !== 'auto') {
            this.showToast('Please switch to Auto mode first', 'error');
            return;
        }
        
        this.sendCommand({ cmd: 'start' });
        this.showToast('Starting automatic mode...', 'info');
    }
    
    save() {
        if (!this.connected) {
            this.showToast('Please connect first', 'error');
            return;
        }
        
        // Prompt for pose name
        const poseName = prompt('Enter a name for this pose (optional):', '');
        
        // Save current pose
        const pose = { ...this.servoValues };
        this.sendCommand({ 
            cmd: 'save',
            pose: pose,
            name: poseName || undefined
        });
        
        // Visual feedback
        const btn = document.getElementById('btnSave');
        if (btn) {
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i><span>Saving...</span>';
            btn.style.background = 'var(--success)';
            
            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.style.background = '';
            }, 2000);
        }
    }
    
    reset() {
        if (!this.connected) {
            this.showToast('Please connect first', 'error');
            return;
        }
        
        if (!confirm('Reset arm to home position (90° all joints, claw closed)?')) {
            return;
        }
        
        this.sendCommand({ cmd: 'reset' });
        this.showToast('Resetting arm to home position...', 'info');
        
        // Reset sliders to home position immediately for visual feedback
        this.resetSliders();
    }
    
    resetSliders() {
        const homeValues = {
            base: 90,
            forearm: 90,
            arm: 90,
            wrist_yaw: 90,
            wrist_pitch: 90,
            claw: 0,
            speed: 50
        };
        
        Object.keys(homeValues).forEach(servoName => {
            const slider = document.getElementById(`${servoName}Slider`);
            if (slider) {
                slider.value = homeValues[servoName];
                this.servoValues[servoName] = homeValues[servoName];
                this.updateServoValue(servoName, homeValues[servoName]);
                this.updateSliderFill(servoName, slider);
            }
        });
    }
    
    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connectionStatus');
        const statusText = statusEl.querySelector('.status-text');
        
        if (connected) {
            statusEl.classList.remove('disconnected');
            statusEl.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusEl.classList.remove('connected');
            statusEl.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
        }
        
        // Update button states
        const btnConnect = document.getElementById('btnConnect');
        const btnDisconnect = document.getElementById('btnDisconnect');
        const btnStart = document.getElementById('btnStart');
        const btnSave = document.getElementById('btnSave');
        const btnReset = document.getElementById('btnReset');
        
        if (btnConnect) btnConnect.disabled = connected;
        if (btnDisconnect) btnDisconnect.disabled = !connected;
        if (btnStart) btnStart.disabled = !connected;
        if (btnSave) btnSave.disabled = !connected;
        if (btnReset) btnReset.disabled = !connected;
    }
    
    updateMode() {
        const modeValue = document.getElementById('modeValue');
        if (modeValue) {
            modeValue.textContent = this.mode === 'auto' ? 'Automatic' : 'Manual';
        }
        
        // Disable servo sliders in auto mode (but keep speed enabled)
        const sliders = document.querySelectorAll('.servo-slider');
        sliders.forEach(slider => {
            const servoName = slider.dataset.servo;
            if (servoName !== 'speed') {
                slider.disabled = this.mode === 'auto';
                slider.style.opacity = this.mode === 'auto' ? '0.5' : '1';
            } else {
                // Speed slider is always enabled
                slider.disabled = false;
                slider.style.opacity = '1';
            }
        });
        
        // Send mode change
        if (this.connected) {
            this.sendCommand({
                cmd: 'set_mode',
                mode: this.mode
            });
        }
    }
    
    handleTelemetry(data) {
        // Update distance
        if (data.distance_mm !== undefined) {
            const distanceEl = document.getElementById('distanceValue');
            if (distanceEl) {
                if (data.distance_mm > 0) {
                    distanceEl.textContent = `${data.distance_mm} mm`;
                } else {
                    distanceEl.textContent = '-- mm';
                }
            }
        }
        
        // Update status
        if (data.status) {
            const statusEl = document.getElementById('statusValue');
            if (statusEl) {
                const statusText = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                statusEl.textContent = statusText;
                
                // Color code status
                const statusColors = {
                    'idle': '#6c757d',
                    'connected': '#28a745',
                    'disconnected': '#dc3545',
                    'unavailable': '#ffc107',
                    'moving': '#007bff',
                    'running': '#007bff',
                    'picking': '#ffc107',
                    'error': '#dc3545',
                    'ready': '#28a745'
                };
                
                statusEl.style.color = statusColors[data.status.toLowerCase()] || '#2c3e50';
            }
        }
        
        // Update mode
        if (data.mode) {
            const modeEl = document.getElementById('modeValue');
            if (modeEl) {
                modeEl.textContent = data.mode === 'auto' ? 'Automatic' : 'Manual';
            }
        }
        
        // Update connection status if provided
        if (data.arduino_connected !== undefined) {
            this.updateConnectionStatus(data.arduino_connected);
        }
        
        // Update last update time
        const lastUpdateEl = document.getElementById('lastUpdate');
        if (lastUpdateEl) {
            const now = new Date();
            lastUpdateEl.textContent = now.toLocaleTimeString();
        }
    }
    
    handleStatus(data) {
        console.log('Status update:', data);
        
        if (data.message) {
            console.log('Status:', data.message);
            
            // Show visual feedback for important status messages
            if (data.connected !== undefined) {
                this.updateConnectionStatus(data.connected);
            }
            
            // Show toast notification for important messages
            if (data.message.includes('error') || data.message.includes('failed')) {
                this.showToast(data.message, 'error');
            } else if (data.message.includes('saved') || data.message.includes('success')) {
                this.showToast(data.message, 'success');
            }
        }
    }
    
    showToast(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            background: ${type === 'error' ? '#dc3545' : type === 'success' ? '#28a745' : '#007bff'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    updateTelemetry() {
        // Initial telemetry update
        const lastUpdateEl = document.getElementById('lastUpdate');
        if (lastUpdateEl) {
            lastUpdateEl.textContent = '--';
        }
    }
}

// Initialize controller when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Robotic Arm Controller...');
    try {
        window.armController = new RoboticArmController();
        console.log('Controller initialized successfully');
    } catch (error) {
        console.error('Failed to initialize controller:', error);
    }
});

