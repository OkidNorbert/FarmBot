# File Organization Recommendations

## Current Status
After cleanup, the project has **22 Python files** in the root directory. Here are recommendations for better organization:

## Recommended Organization

### 1. **Legacy/Unused Files (Can Remove or Archive)**

#### `arduino_servo.ino` - **OUTDATED**
- **Status**: Old simple 3-servo controller
- **Replacement**: `arduino/main_firmware/main_firmware.ino` (comprehensive 6-DOF system)
- **Action**: Move to `arduino/legacy/` or delete (only mentioned in docs, not used)

#### `pi_web_interface.py` - **POTENTIALLY REDUNDANT**
- **Status**: Separate Pi-specific web interface
- **Replacement**: `web_interface.py` (unified interface for both regular and Pi)
- **Action**: Verify if still needed, or merge functionality into `web_interface.py`

### 2. **Test Files (Move to `tests/` directory)**

Currently in root, should be in `tests/`:
- `test_arduino_ble.py` → `tests/test_arduino_ble.py`
- `test_camera.py` → `tests/test_camera.py`
- `check_compatibility.py` → `tests/check_compatibility.py`
- `verify_fixes.py` → `tests/verify_fixes.py`

### 3. **Utility Scripts (Create `scripts/` directory)**

Create `scripts/` directory for utility scripts:

**Dataset Utilities:**
- `extract_dataset.py` → `scripts/extract_dataset.py`
- `prepare_multi_tomato_dataset.py` → `scripts/prepare_multi_tomato_dataset.py`

**Calibration Utilities:**
- `coordinate_mapper.py` → `scripts/coordinate_mapper.py`
- `calibrate_homography.py` → `scripts/calibrate_homography.py`

**Learning Utilities:**
- `continuous_learning.py` → `scripts/continuous_learning.py`

**Diagnostic Utilities:**
- `check_bluetooth.sh` → `scripts/check_bluetooth.sh`

### 4. **Training Scripts (Create `scripts/training/` subdirectory)**

- `train_tomato_classifier.py` → `scripts/training/train_tomato_classifier.py`
- `train_yolo.py` → `scripts/training/train_yolo.py`
- `auto_train.py` → `scripts/training/auto_train.py`

### 5. **Inference Scripts (Create `scripts/inference/` subdirectory)**

- `inference_classifier.py` → `scripts/inference/inference_classifier.py`
- `inference_pi.py` → `scripts/inference/inference_pi.py`

### 6. **Configuration Files (Create `config/` directory)**

Runtime configuration/data files:
- `calibration_data.json` → `config/calibration_data.json`
- `camera_preference.json` → `config/camera_preference.json`
- `monitoring_stats.json` → `config/monitoring_stats.json`
- `detection_log.csv` → `config/detection_log.csv`
- `data.yaml` → `config/data.yaml` (or keep in root if used by multiple scripts)

### 7. **Startup Scripts (Keep in root or create `scripts/startup/`)**

**Keep in root** (commonly used):
- `start.sh` - Main startup script
- `setup.sh` - Setup script

**Move to `scripts/startup/`:**
- `start_web_interface.py` → `scripts/startup/start_web_interface.py`
- `start_web_port.py` → `scripts/startup/start_web_port.py`
- `pi_startup.sh` → `scripts/startup/pi_startup.sh`
- `deploy_to_pi.sh` → `scripts/deployment/deploy_to_pi.sh`

### 8. **Service Files (Keep in root)**

- `tomato_sorter.service` - Systemd service file (needs to be in root for easy access)

### 9. **Core Application Files (Keep in root)**

These are the main entry points and should stay in root:
- `web_interface.py` - Main web application
- `hardware_controller.py` - Hardware abstraction layer
- `pi_controller.py` - Pi-specific controller (if still needed)
- `yolo_service.py` - YOLO detection service
- `ik_solver.py` - Inverse kinematics solver

## Proposed Directory Structure

```
emebeded/
├── README.md
├── PROJECT_README.md
├── COMMISSIONING_CHECKLIST.md
├── FILES_TO_REMOVE.md
├── requirements.txt
├── setup.sh
├── start.sh
├── tomato_sorter.service
│
├── web_interface.py          # Main web app (KEEP IN ROOT)
├── hardware_controller.py    # Hardware abstraction (KEEP IN ROOT)
├── pi_controller.py          # Pi controller (KEEP IN ROOT)
├── yolo_service.py           # YOLO service (KEEP IN ROOT)
├── ik_solver.py              # IK solver (KEEP IN ROOT)
│
├── config/                   # Configuration files
│   ├── calibration_data.json
│   ├── camera_preference.json
│   ├── monitoring_stats.json
│   ├── detection_log.csv
│   └── data.yaml
│
├── scripts/                  # Utility scripts
│   ├── training/
│   │   ├── train_tomato_classifier.py
│   │   ├── train_yolo.py
│   │   └── auto_train.py
│   ├── inference/
│   │   ├── inference_classifier.py
│   │   └── inference_pi.py
│   ├── calibration/
│   │   ├── coordinate_mapper.py
│   │   └── calibrate_homography.py
│   ├── dataset/
│   │   ├── extract_dataset.py
│   │   └── prepare_multi_tomato_dataset.py
│   ├── startup/
│   │   ├── start_web_interface.py
│   │   ├── start_web_port.py
│   │   └── pi_startup.sh
│   ├── deployment/
│   │   └── deploy_to_pi.sh
│   ├── continuous_learning.py
│   └── check_bluetooth.sh
│
├── tests/                    # Test files
│   ├── test_websocket.py
│   ├── test_arduino_ble.py
│   ├── test_camera.py
│   ├── check_compatibility.py
│   └── verify_fixes.py
│
├── arduino/                  # Arduino firmware
│   ├── main_firmware/
│   ├── legacy/               # Old firmware (if keeping)
│   │   └── arduino_servo.ino
│   └── README.md
│
├── calibration/              # Calibration tools (existing)
│   └── pixel_to_servo_wizard.py
│
├── docs/                     # Documentation (existing)
├── templates/                # Web templates (existing)
├── static/                   # Static files (existing)
├── models/                   # AI models (existing)
├── datasets/                 # Datasets (existing)
├── hardware/                 # Hardware docs (existing)
├── web/                      # Web API docs (existing)
└── ... (other existing directories)
```

## Priority Actions

### High Priority (Recommended)
1. ✅ **Move test files to `tests/`** - Better organization
2. ✅ **Create `config/` directory** - Organize runtime config files
3. ✅ **Move utility scripts to `scripts/`** - Clean up root directory

### Medium Priority (Optional)
4. ⚠️ **Archive `arduino_servo.ino`** - Old firmware, not used
5. ⚠️ **Review `pi_web_interface.py`** - Check if still needed or merge

### Low Priority (Nice to Have)
6. 💡 **Organize training/inference scripts** - Subdirectories for better categorization
7. 💡 **Move startup scripts** - Only if you want cleaner root

## Notes

- **Keep core application files in root** - These are entry points and should be easily accessible
- **Configuration files** - Consider if they need to be in root for easy access, or can be in `config/`
- **Scripts** - Moving to `scripts/` makes them organized but requires updating any references
- **Tests** - Moving to `tests/` is standard practice and improves organization

## Implementation

If you want to proceed with organization, I can:
1. Create the new directory structure
2. Move files to appropriate locations
3. Update any import paths or references
4. Update documentation to reflect new structure

Would you like me to proceed with any of these organizational changes?

