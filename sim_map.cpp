#include <iostream>

long map_func(long x, long in_min, long in_max, long out_min, long out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int constrain(int x, int a, int b) {
    if (x < a) return a;
    if (x > b) return b;
    return x;
}

int widthToGripAngle(int width_mm) {
    int target_width = width_mm - 6; 
    int angle = map_func(target_width, 10, 70, 110, 35);
    return constrain(angle, 30, 110);
}

int widthToOpenAngle(int width_mm) {
    int grip = widthToGripAngle(width_mm);
    int open = grip - 30;
    if (open < 30) open = 30;
    return open;
}

int main() {
    std::cout << "Width 30 -> Grip: " << widthToGripAngle(30) << " Open: " << widthToOpenAngle(30) << std::endl;
    std::cout << "Width 60 -> Grip: " << widthToGripAngle(60) << " Open: " << widthToOpenAngle(60) << std::endl;
    return 0;
}
