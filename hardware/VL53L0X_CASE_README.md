# VL53L0X V2 TOF Sensor 3D Printable Case

OpenSCAD designs for 3D printable cases/holders for the VL53L0X V2 Time-of-Flight sensor with soldered pins.

## 🎨 Three Creative Designs Available!

### 1. **Open-Top Design** (`vl53l0x_case.scad`) ⭐ RECOMMENDED
   - **Easy insertion**: Sensor drops in from the top - super simple!
   - **Open top**: No complex maneuvering needed
   - **Retention tabs**: Flexible tabs hold sensor in place
   - **Rounded corners**: Professional look
   - **Best for**: Most users, easiest to use

### 2. **Slide-In Design** (`vl53l0x_case_slide.scad`)
   - **Side insertion**: Sensor slides in from the side
   - **Guide rails**: Helps align sensor during insertion
   - **Flexible clips**: Snap into place
   - **Best for**: Tight spaces, horizontal mounting

### 3. **Clamshell Design** (`vl53l0x_case_clamshell.scad`) 🎯 MOST CREATIVE
   - **Two-part design**: Base and top separate
   - **Drop-in placement**: Sensor simply sits in base
   - **Top clips on**: Easy assembly, no insertion needed
   - **Easy maintenance**: Remove top to access sensor
   - **Best for**: Frequent access, professional projects

## 📋 Features (All Designs)

- **Secure PCB holder** - Snug fit for the sensor board
- **Pin accommodation** - Slots/channels for soldered pins
- **Lens opening** - Clear opening for the sensor lens
- **Mounting holes** - M3 mounting holes for attachment
- **3D printable** - No supports required, prints flat on bed

## 📐 Dimensions

- **PCB Size**: 20mm × 12.7mm (0.8" × 0.5")
- **Pin Spacing**: 2.54mm (standard 0.1" spacing)
- **Sensor Lens**: 4.5mm diameter
- **Mounting Holes**: M3 (3mm diameter, 15mm spacing)

## 🖨️ Printing Instructions

### Recommended Settings:
- **Layer Height**: 0.2mm
- **Infill**: 20-30%
- **Supports**: None required
- **Orientation**: Print flat on bed (base down)
- **Material**: PLA or PETG recommended

### Print Settings:
```
Layer Height: 0.2mm
First Layer Height: 0.3mm
Wall Thickness: 2.0mm
Top/Bottom Layers: 3-4
Infill: 20-30%
Print Speed: 50-60mm/s
Bed Temperature: 60°C (PLA) / 80°C (PETG)
Nozzle Temperature: 210°C (PLA) / 240°C (PETG)
```

## 🔧 Customization

Edit the parameters at the top of `vl53l0x_case.scad`:

```openscad
// Adjust these values if your sensor dimensions differ
pcb_length = 20.0;      // Length of PCB (mm)
pcb_width = 12.7;       // Width of PCB (mm)
pin_length = 10.0;      // Length of pins below PCB (mm)
lens_diameter = 4.5;    // Sensor lens diameter (mm)
wall_thickness = 2.0;   // Wall thickness (mm)
```

### To Enable/Disable Features:
- **Snap-fit tabs**: Set `snap_fit_enabled = false;` to disable
- **Lid**: Uncomment the lid module at the bottom to render separately

## 📦 Assembly Instructions

### Open-Top Design:
1. **Print the case** - Print with base flat on bed
2. **Drop in sensor** - Simply place VL53L0X sensor into the open top
   - PCB sits on the base platform
   - Sensor lens faces forward
   - Pins point backward through slots
3. **Pins route automatically** - Pins pass through slots in back wall
4. **Retention tabs hold it** - Flexible tabs keep sensor in place
5. **Mount** - Use M3 screws through mounting holes

### Slide-In Design:
1. **Print the case** - Print with base flat on bed
2. **Slide sensor in** - Insert sensor from the side opening
3. **Guide rails help** - Rails align sensor during insertion
4. **Clips snap in** - Retention clips hold sensor
5. **Mount** - Use M3 screws

### Clamshell Design:
1. **Print both parts** - Base and top print separately
2. **Place sensor in base** - Sensor simply sits in the base cavity
3. **Clip on top** - Top part clips onto alignment posts
4. **Snap tabs secure** - Tabs hold everything together
5. **Mount** - Use M3 screws through both parts

## 🔌 Pin Configuration

The case accommodates 4 pins (standard configuration):
- **VCC** (Power)
- **GND** (Ground)
- **SDA** (I2C Data)
- **SCL** (I2C Clock)

Pins are spaced at 2.54mm (0.1") standard spacing.

## 📍 Mounting Options

### Option 1: Screw Mounting
- Use M3 screws through the mounting holes
- 15mm spacing between holes
- Suitable for permanent mounting

### Option 2: Adhesive Mounting
- Remove mounting hole cylinders in OpenSCAD
- Use double-sided tape or adhesive
- Lighter weight option

### Option 3: Snap-fit
- Enable snap-fit tabs in the code
- Provides secure retention without screws
- Easy removal for maintenance

## 🎨 Design Variants

### Variant 1: Low Profile
Reduce `top_clearance` and `wall_thickness` for a more compact design.

### Variant 2: High Clearance
Increase `pin_length` and `top_clearance` for longer pins or additional components.

### Variant 3: With Lid
Uncomment the lid module to print a protective cover.

## 🔍 Troubleshooting

### Sensor doesn't fit:
- Check PCB dimensions match your sensor
- Adjust `pcb_length` and `pcb_width` parameters
- Increase `side_clearance` slightly

### Pins don't fit:
- Measure your actual pin spacing
- Adjust `pin_spacing` if different from 2.54mm
- Increase `pin_diameter` tolerance

### Lens blocked:
- Increase `lens_diameter` parameter
- Check lens position on your sensor board
- Adjust lens opening position if needed

## 📝 Notes

- The design assumes standard VL53L0X module dimensions
- Some modules may have slightly different dimensions
- Always verify dimensions before printing
- Test fit with a small test print first

## 🔗 Related Files

- `vl53l0x_case.scad` - Main OpenSCAD design file
- Arduino test code: `arduino/test_tof/test_tof.ino`

## 📄 License

This design is provided as-is for use with the AI Tomato Sorter project.

