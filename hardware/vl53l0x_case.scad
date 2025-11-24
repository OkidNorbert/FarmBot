/*
 * VL53L0X V2 TOF Sensor Case - Open Design
 * 3D Printable holder for VL53L0X sensor with soldered pins
 * 
 * Creative open design with easy sensor insertion
 * - Open top for easy sensor placement
 * - Side slots for pin access
 * - Snap-fit retention tabs
 * - Clear visual guide for sensor placement
 */

// ==========================================
// CONFIGURATION PARAMETERS
// ==========================================

// PCB dimensions
pcb_length = 20.0;      // Length of PCB (mm)
pcb_width = 12.7;       // Width of PCB (mm)
pcb_thickness = 1.6;    // PCB thickness (mm)

// Pin configuration
pin_spacing = 2.54;     // Standard pin spacing (mm)
pin_count = 4;          // Number of pins (VCC, GND, SDA, SCL)
pin_diameter = 0.8;     // Pin diameter (mm)
pin_length = 10.0;      // Length of pins below PCB (mm)

// Sensor lens
lens_diameter = 4.5;    // Sensor lens diameter (mm)
lens_height = 1.0;      // Lens protrusion above PCB (mm)

// Case dimensions
wall_thickness = 2.0;   // Wall thickness (mm)
base_thickness = 2.5;   // Base thickness (mm)
side_wall_height = 3.0; // Height of side walls (mm)
corner_radius = 2.0;    // Rounded corners (mm)

// Clearances
pcb_clearance = 0.3;    // Clearance around PCB for easy insertion
pin_clearance = 0.5;    // Clearance around pins

// Mounting holes
mount_hole_diameter = 3.0;  // M3 screw hole
mount_hole_spacing = 15.0;   // Distance between mounting holes

// Retention tabs
tab_width = 3.0;        // Width of retention tabs
tab_thickness = 0.8;   // Thickness of tabs
tab_height = 1.5;      // Height tabs extend above PCB

// ==========================================
// CALCULATED DIMENSIONS
// ==========================================

total_length = pcb_length + (pcb_clearance * 2) + (wall_thickness * 2);
total_width = pcb_width + (pcb_clearance * 2) + (wall_thickness * 2);
base_height = base_thickness + pcb_thickness + pin_length;

// Pin positions (centered on PCB, along one edge)
pin_start_x = -(pin_count - 1) * pin_spacing / 2;
pin_y_position = -pcb_width / 2 - 1.5; // Pins on bottom edge

// ==========================================
// MAIN CASE BASE
// ==========================================

module rounded_rect(size, r) {
    translate([-size[0]/2, -size[1]/2, 0])
        minkowski() {
            cube([size[0] - r*2, size[1] - r*2, size[2]]);
            cylinder(r=r, h=0.01, $fn=16);
        }
}

module case_base() {
    // Main base with rounded corners
    rounded_rect([total_length, total_width, base_height], corner_radius);
    
    // Side walls (partial, not full height)
    translate([-total_length/2, -total_width/2, base_height])
        cube([total_length, wall_thickness, side_wall_height]);
    translate([-total_length/2, total_width/2 - wall_thickness, base_height])
        cube([total_length, wall_thickness, side_wall_height]);
    
    // Back wall (where pins exit)
    translate([-total_length/2, -total_width/2, base_height])
        cube([wall_thickness, total_width, side_wall_height]);
    
    // Front wall (partial, with lens opening)
    front_wall_width = total_width - lens_diameter - 4;
    translate([total_length/2 - wall_thickness, -front_wall_width/2, base_height])
        cube([wall_thickness, front_wall_width, side_wall_height]);
}

// ==========================================
// CUTOUTS AND OPENINGS
// ==========================================

module case_cutouts() {
    // PCB cavity (open top for easy insertion)
    translate([
        -pcb_length/2 - pcb_clearance,
        -pcb_width/2 - pcb_clearance,
        base_thickness
    ])
        cube([
            pcb_length + (pcb_clearance * 2),
            pcb_width + (pcb_clearance * 2),
            pcb_thickness + 5
        ]);
    
    // Sensor lens opening (front)
    translate([
        total_length/2 - wall_thickness - 1,
        0,
        base_thickness + pcb_thickness
    ])
        rotate([0, 90, 0])
            cylinder(h = wall_thickness + 2, d = lens_diameter + 1, $fn=32);
    
    // Pin exit slots (back wall)
    for (i = [0:pin_count-1]) {
        translate([
            -total_length/2 - 1,
            pin_y_position + (i * pin_spacing),
            base_thickness
        ])
            cube([
                wall_thickness + 2,
                pin_diameter + pin_clearance,
                pin_length + 5
            ]);
    }
    
    // Mounting holes
    translate([mount_hole_spacing/2, 0, -1])
        cylinder(h = base_thickness + 2, d = mount_hole_diameter, $fn=16);
    translate([-mount_hole_spacing/2, 0, -1])
        cylinder(h = base_thickness + 2, d = mount_hole_diameter, $fn=16);
    
    // Wire routing channel (optional)
    translate([0, -total_width/2 - 1, base_thickness + pcb_thickness])
        cube([10, 4, 6], center=true);
}

// ==========================================
// RETENTION TABS
// ==========================================

module retention_tabs() {
    // Front tabs (hold PCB in place)
    tab_positions = [
        [pcb_length/3, pcb_width/2 + pcb_clearance],
        [-pcb_length/3, pcb_width/2 + pcb_clearance]
    ];
    
    for (pos = tab_positions) {
        translate([pos[0], pos[1], base_thickness + pcb_thickness])
            rotate([90, 0, 0])
                linear_extrude(height = tab_thickness)
                    polygon([
                        [-tab_width/2, 0],
                        [tab_width/2, 0],
                        [tab_width/2, tab_height],
                        [0, tab_height + 1],
                        [-tab_width/2, tab_height]
                    ]);
    }
    
    // Side tabs
    translate([pcb_length/2 + pcb_clearance, 0, base_thickness + pcb_thickness])
        rotate([0, 90, 0])
            linear_extrude(height = tab_thickness)
                polygon([
                    [-tab_width/2, 0],
                    [tab_width/2, 0],
                    [tab_width/2, tab_height],
                    [0, tab_height + 1],
                    [-tab_width/2, tab_height]
                ]);
}

// ==========================================
// VISUAL GUIDES (optional, for reference)
// ==========================================

module pcb_outline() {
    %translate([-pcb_length/2, -pcb_width/2, base_thickness])
        color("green", 0.3)
            cube([pcb_length, pcb_width, pcb_thickness]);
    
    %translate([0, pcb_width/2 + 1, base_thickness + pcb_thickness])
        color("red", 0.5)
            cylinder(h = lens_height, d = lens_diameter, $fn=32);
}

// ==========================================
// MAIN CASE ASSEMBLY
// ==========================================

module vl53l0x_case() {
    difference() {
        case_base();
        case_cutouts();
    }
    
    // Add retention tabs
    retention_tabs();
    
    // Visual guide (comment out for final print)
    // pcb_outline();
}

// ==========================================
// RENDER
// ==========================================

vl53l0x_case();

// ==========================================
// INSTRUCTIONS
// ==========================================
/*
 * ASSEMBLY INSTRUCTIONS:
 * 
 * 1. Print the case with base flat on bed
 * 2. The case has an OPEN TOP - sensor slides in from above
 * 3. Place VL53L0X sensor into the cavity:
 *    - PCB sits on the base platform
 *    - Sensor lens faces forward (toward lens opening)
 *    - Pins point backward (toward pin slots)
 * 4. Pins will pass through slots in the back wall
 * 5. Retention tabs will hold the PCB in place
 * 6. Use M3 screws through mounting holes to attach
 * 
 * The design is intentionally open on top for easy insertion!
 */
