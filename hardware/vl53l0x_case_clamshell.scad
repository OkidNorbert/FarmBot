/*
 * VL53L0X V2 TOF Sensor Case - Clamshell Design
 * Two-part clamshell design - most creative and user-friendly!
 * 
 * Features:
 * - Two-part design: base and top
 * - Sensor simply sits in base, top clips on
 * - No complex insertion needed
 * - Easy to remove for maintenance
 */

// ==========================================
// CONFIGURATION
// ==========================================

pcb_length = 20.0;
pcb_width = 12.7;
pcb_thickness = 1.6;
pin_spacing = 2.54;
pin_count = 4;
pin_diameter = 0.8;
pin_length = 10.0;
lens_diameter = 4.5;
wall_thickness = 2.0;
base_thickness = 2.5;
top_thickness = 1.5;
clearance = 0.3;
corner_radius = 2.0;

// ==========================================
// HELPER MODULES
// ==========================================

module rounded_square(size, r) {
    translate([-size[0]/2, -size[1]/2, 0])
        minkowski() {
            cube([size[0] - r*2, size[1] - r*2, size[2]]);
            cylinder(r=r, h=0.01, $fn=16);
        }
}

// ==========================================
// BASE PART
// ==========================================

module base_part() {
    total_l = pcb_length + clearance * 2 + wall_thickness * 2;
    total_w = pcb_width + clearance * 2 + wall_thickness * 2;
    base_h = base_thickness + pcb_thickness + pin_length;
    
    difference() {
        // Base body
        rounded_square([total_l, total_w, base_h], corner_radius);
        
        // PCB cavity
        translate([
            -pcb_length/2 - clearance,
            -pcb_width/2 - clearance,
            base_thickness
        ])
            cube([
                pcb_length + clearance * 2,
                pcb_width + clearance * 2,
                pcb_thickness + 0.5
            ]);
        
        // Pin channels (through base)
        for (i = [0:pin_count-1]) {
            translate([
                -(pin_count-1) * pin_spacing/2 + i * pin_spacing,
                0,
                base_thickness
            ])
                cylinder(h = pin_length + 2, d = pin_diameter + 0.5, $fn=16);
        }
        
        // Lens opening (front)
        translate([
            total_l/2 - wall_thickness - 1,
            0,
            base_thickness + pcb_thickness
        ])
            rotate([0, 90, 0])
                cylinder(h = wall_thickness + 2, d = lens_diameter + 1, $fn=32);
        
        // Mounting holes
        translate([15/2, 0, -1])
            cylinder(h = base_thickness + 2, d = 3, $fn=16);
        translate([-15/2, 0, -1])
            cylinder(h = base_thickness + 2, d = 3, $fn=16);
    }
    
    // Alignment posts (for top part)
    post_d = 2.0;
    post_h = 2.0;
    translate([total_l/2 - 3, total_w/2 - 3, base_h])
        cylinder(h = post_h, d = post_d, $fn=16);
    translate([-total_l/2 + 3, total_w/2 - 3, base_h])
        cylinder(h = post_h, d = post_d, $fn=16);
    translate([total_l/2 - 3, -total_w/2 + 3, base_h])
        cylinder(h = post_h, d = post_d, $fn=16);
    translate([-total_l/2 + 3, -total_w/2 + 3, base_h])
        cylinder(h = post_h, d = post_d, $fn=16);
}

// ==========================================
// TOP PART (LID)
// ==========================================

module top_part() {
    total_l = pcb_length + clearance * 2 + wall_thickness * 2;
    total_w = pcb_width + clearance * 2 + wall_thickness * 2;
    
    difference() {
        // Top body
        rounded_square([total_l, total_w, top_thickness], corner_radius);
        
        // Lens opening
        translate([0, 0, -1])
            cylinder(h = top_thickness + 2, d = lens_diameter + 2, $fn=32);
        
        // Alignment holes
        post_d = 2.2; // Slightly larger for fit
        translate([total_l/2 - 3, total_w/2 - 3, -1])
            cylinder(h = top_thickness + 2, d = post_d, $fn=16);
        translate([-total_l/2 + 3, total_w/2 - 3, -1])
            cylinder(h = top_thickness + 2, d = post_d, $fn=16);
        translate([total_l/2 - 3, -total_w/2 + 3, -1])
            cylinder(h = top_thickness + 2, d = post_d, $fn=16);
        translate([-total_l/2 + 3, -total_w/2 + 3, -1])
            cylinder(h = top_thickness + 2, d = post_d, $fn=16);
        
        // Mounting holes
        translate([15/2, 0, -1])
            cylinder(h = top_thickness + 2, d = 3, $fn=16);
        translate([-15/2, 0, -1])
            cylinder(h = top_thickness + 2, d = 3, $fn=16);
    }
    
    // Retention tabs (snap into place)
    tab_w = 3.0;
    tab_t = 0.8;
    tab_h = 1.5;
    
    // Front tabs
    translate([pcb_length/3, total_w/2 - tab_t, top_thickness])
        cube([tab_w, tab_t, tab_h], center=true);
    translate([-pcb_length/3, total_w/2 - tab_t, top_thickness])
        cube([tab_w, tab_t, tab_h], center=true);
}

// ==========================================
// RENDER
// ==========================================

// Render both parts side by side for printing
base_part();

translate([0, pcb_width + 15, 0])
    top_part();

// Visual guide (uncomment to see sensor position)
/*
%translate([-pcb_length/2, -pcb_width/2, base_thickness])
    color("green", 0.3)
        cube([pcb_length, pcb_width, pcb_thickness]);
*/

