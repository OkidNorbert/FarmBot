# Commissioning Checklist: Hybrid Tomato-Picking System

## Pre-Commissioning

### Hardware Verification
- [ ] Arduino UNO R4 WiFi board verified working
- [ ] All 6 servos tested individually
- [ ] VL53L0X sensor tested and reading distances
- [ ] Emergency stop switch installed and tested
- [ ] Power supplies verified (Arduino + External servo supply)
- [ ] All ground connections verified (common ground)
- [ ] Decoupling capacitors installed near servo power input

### Software Setup
- [ ] Arduino IDE configured for UNO R4 WiFi
- [ ] Required libraries installed:
  - [ ] Servo
  - [ ] Adafruit_VL53L0X
  - [ ] ArduinoWebsockets
  - [ ] ArduinoJson
  - [ ] WiFiS3 (built-in for R4 WiFi)
- [ ] Web interface dependencies installed
- [ ] YOLO model trained and available

## Phase 1: Basic Hardware Test

### Servo Test
- [ ] Upload `test_all_components.ino` or `diagnose_servos.ino`
- [ ] Verify each servo moves individually
- [ ] Verify all servos home to 90° on startup
- [ ] Test emergency stop functionality
- [ ] Verify servo limits (claw 0-90°, others within safe ranges)

### Sensor Test
- [ ] VL53L0X sensor initializes
- [ ] Distance readings are reasonable (20-1200mm)
- [ ] Sensor responds to objects at various distances

### Communication Test
- [ ] Arduino connects to WiFi network
- [ ] Arduino connects to WebSocket server
- [ ] Telemetry messages received on web interface
- [ ] Commands sent from web interface reach Arduino

## Phase 2: Calibration

### Servo Calibration
- [ ] Physical 90° position verified for each servo
- [ ] Servo trims adjusted if needed
- [ ] Safety limits verified for each joint
- [ ] Calibration data saved to EEPROM

### Bin Position Calibration
- [ ] Right bin (ripe) position recorded
- [ ] Left bin (unripe) position recorded
- [ ] Bin poses saved to EEPROM
- [ ] Test movement to each bin position

### Pixel-to-Robot Mapping
- [ ] Camera mounted and positioned
- [ ] At least 4 calibration points recorded
- [ ] Pixel coordinates mapped to robot positions
- [ ] Calibration table generated and tested

## Phase 3: Motion Testing

### Manual Mode
- [ ] Web interface manual controls work
- [ ] Individual joint control functional
- [ ] Smooth motion verified (no jerky movements)
- [ ] Speed control working

### Pick Sequence Test
- [ ] Approach pose calculation working
- [ ] ToF-based fine positioning working
- [ ] Grasp sequence (claw close) working
- [ ] Lift sequence working
- [ ] Bin routing (ripe/unripe) working
- [ ] Release and return home working

### Safety Tests
- [ ] Emergency stop stops all motion immediately
- [ ] Joint limits prevent over-rotation
- [ ] Timeout handling works (10s state timeout)
- [ ] Low confidence detections rejected

## Phase 4: Integration

### YOLO Integration
- [ ] YOLO service running and detecting tomatoes
- [ ] Detections sent to web backend
- [ ] Web backend converts detections to pick commands
- [ ] Pick commands sent to Arduino via WebSocket

### Automatic Mode
- [ ] AUTO mode enabled via web interface
- [ ] YOLO detections trigger pick sequences
- [ ] System handles multiple detections correctly
- [ ] Pick results logged and displayed

### End-to-End Test
- [ ] Place test object in camera view
- [ ] YOLO detects object
- [ ] Pick command sent to Arduino
- [ ] Arm executes complete pick sequence
- [ ] Object sorted to correct bin
- [ ] System returns to home and ready for next pick

## Phase 5: Field Commissioning

### Environment Setup
- [ ] Camera height and angle adjusted
- [ ] Lighting conditions verified (adequate for detection)
- [ ] Workspace boundaries defined
- [ ] Bin positions physically marked

### Performance Verification
- [ ] Pick success rate > 90%
- [ ] Average pick time < 10 seconds
- [ ] False positive rate < 5%
- [ ] System runs continuously for 30+ minutes without errors

### Safety Verification
- [ ] Emergency stop tested in all scenarios
- [ ] Power loss recovery tested
- [ ] Communication loss recovery tested
- [ ] Operator training completed

## Troubleshooting

### Common Issues

**Servos don't move:**
- Check power supply connections
- Verify ground connections
- Check servo pulse width settings
- Verify pins are correct (D2-D7)

**VL53L0X not working:**
- Check I2C connections (SDA/SCL)
- Verify power (3.3V or 5V)
- Check for I2C address conflicts
- Try different I2C pins if needed

**WiFi/WebSocket not connecting:**
- Verify WiFi credentials in `config.h`
- Check web server IP address
- Verify firewall settings
- Check network connectivity

**Pick sequence fails:**
- Verify ToF sensor readings
- Check joint limits
- Verify bin positions are reachable
- Check for mechanical interference

## Sign-Off

- [ ] All hardware tests passed
- [ ] All calibration completed
- [ ] All safety tests passed
- [ ] End-to-end test successful
- [ ] Documentation reviewed
- [ ] Operator trained
- [ ] System ready for production use

**Commissioned by:** _________________ **Date:** _______________

**Verified by:** _________________ **Date:** _______________

