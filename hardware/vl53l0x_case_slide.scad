/*
 * VL53L0X V2 TOF Sensor Case - Slide-In Design
 * Creative slide-in design for easy sensor insertion
 * 
 * Features:
 * - Sensor slides in from the side
 * - Open top for easy access
 * - Retention clips that snap into place
 * - Clear visual guide
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
side_wall_height = 4.0;
clearance = 0.4;

// ==========================================
// SLIDE-IN CASE
// ==========================================

module slide_case() {
    total_l = pcb_length + clearance * 2 + wall_thickness * 2;
    total_w = pcb_width + clearance * 2 + wall_thickness * 2;
    
    difference() {
        // Main body
        translate([-total_l/2, -total_w/2, 0])
            cube([total_l, total_w, base_thickness + pcb_thickness + pin_length]);
        
        // PCB slot (open on top and one side)
        translate([
            -pcb_length/2 - clearance,
            -pcb_width/2 - clearance - 5, // Extended opening for slide-in
            base_thickness
        ])
            cube([
                pcb_length + clearance * 2,
                pcb_width + clearance * 2 + 10, // Extra space for insertion
                pcb_thickness + 1
            ]);
        
        // Pin channels
        for (i = [0:pin_count-1]) {
            translate([
                -(pin_count-1) * pin_spacing/2 + i * pin_spacing,
                -total_w/2 - 1,
                base_thickness
            ])
                cube([pin_diameter + 0.5, pin_length + 5, pin_length + 5]);
        }
        
        // Lens opening
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
    
    // Retention clips (flexible tabs)
    clip_width = 2.0;
    clip_thickness = 0.6;
    
    // Front clip
    translate([
        pcb_length/3,
        pcb_width/2 + clearance,
        base_thickness + pcb_thickness
    ])
        rotate([90, 0, 0])
            linear_extrude(height = clip_thickness)
                polygon([
                    [-clip_width/2, 0],
                    [clip_width/2, 0],
                    [clip_width/2, 2],
                    [0, 2.5],
                    [-clip_width/2, 2]
                ]);
    
    translate([
        -pcb_length/3,
        pcb_width/2 + clearance,
        base_thickness + pcb_thickness
    ])
        rotate([90, 0, 0])
            linear_extrude(height = clip_thickness)
                polygon([
                    [-clip_width/2, 0],
                    [clip_width/2, 0],
                    [clip_width/2, 2],
                    [0, 2.5],
                    [-clip_width/2, 2]
                ]);
    
    // Side guide rails
    translate([
        -pcb_length/2 - clearance - 0.5,
        -pcb_width/2 - clearance,
        base_thickness + pcb_thickness - 0.5
    ])
        cube([1, pcb_width + clearance * 2, 1]);
}

slide_case();

// Visual guide
%translate([-pcb_length/2, -pcb_width/2, base_thickness])
    color("green", 0.3)
        cube([pcb_length, pcb_width, pcb_thickness]);

