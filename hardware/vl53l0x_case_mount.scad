/*
 * VL53L0X V2 TOF Sensor Case with Mounting Bracket
 * Alternative design with integrated mounting bracket
 * 
 * This variant includes a mounting bracket for attachment to robotic arm
 * 
 * Note: This is a standalone version. Copy the vl53l0x_case() module
 * from vl53l0x_case.scad or include it using: include <vl53l0x_case.scad>
 */

// Include the main case design
include <vl53l0x_case.scad>

// ==========================================
// MOUNTING BRACKET CONFIGURATION
// ==========================================

bracket_length = 30.0;      // Length of mounting bracket
bracket_width = 20.0;        // Width of mounting bracket
bracket_thickness = 3.0;     // Thickness of bracket
bracket_hole_diameter = 4.0; // M4 mounting hole
bracket_hole_spacing = 20.0; // Spacing between bracket holes

// ==========================================
// MOUNTING BRACKET
// ==========================================

module mounting_bracket() {
    difference() {
        // Bracket base
        translate([-bracket_length/2, -bracket_width/2, -bracket_thickness])
            cube([bracket_length, bracket_width, bracket_thickness]);
        
        // Mounting holes
        translate([bracket_hole_spacing/2, 0, -bracket_thickness - 1])
            cylinder(h = bracket_thickness + 2, d = bracket_hole_diameter, $fn=16);
        translate([-bracket_hole_spacing/2, 0, -bracket_thickness - 1])
            cylinder(h = bracket_thickness + 2, d = bracket_hole_diameter, $fn=16);
    }
}

// ==========================================
// COMBINED CASE WITH BRACKET
// ==========================================

module vl53l0x_case_with_mount() {
    // Main case
    vl53l0x_case();
    
    // Mounting bracket
    mounting_bracket();
}

// ==========================================
// RENDER
// ==========================================

vl53l0x_case_with_mount();

