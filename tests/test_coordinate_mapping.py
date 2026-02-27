import numpy as np
import cv2
import math

class MockHardwareController:
    def __init__(self):
        self.homography_matrix = None
        self.logger = type('MockLogger', (), {'debug': print, 'error': print, 'info': print})()

    def pixel_to_arm_coordinates(self, pixel_x, pixel_y):
        try:
            if self.homography_matrix is not None:
                pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
                arm_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
                arm_x = float(arm_point[0][0][0])
                arm_y = float(arm_point[0][0][1])
                return arm_x, arm_y
            else:
                scale_x = 250.0 / 640.0
                scale_y = 180.0 / 480.0
                arm_x = pixel_x * scale_x - 125
                arm_y = pixel_y * scale_y + 80
                return arm_x, arm_y
        except Exception as e:
            return None

def test_coordinate_mapping():
    hw = MockHardwareController()
    
    print("Testing Fallback Mapping...")
    # Center pixel (320, 240) should be roughly (0, 170) in mm with new fallback
    x, y = hw.pixel_to_arm_coordinates(320, 240)
    print(f"Pixel(320, 240) -> Arm({x:.1f}, {y:.1f})")
    assert abs(x - 0) < 1.0
    assert abs(y - 170) < 1.0
    
    print("\nTesting Homography Mapping...")
    # Create a simple mapping: 1px = 1mm, offset by (10, 20)
    src_pts = np.array([[0,0], [100,0], [100,100], [0,100]], dtype=np.float32)
    dst_pts = np.array([[10,20], [110,20], [110,120], [10,120]], dtype=np.float32)
    h, _ = cv2.findHomography(src_pts, dst_pts)
    hw.homography_matrix = h
    
    x, y = hw.pixel_to_arm_coordinates(50, 50)
    print(f"Pixel(50, 50) -> Arm({x:.1f}, {y:.1f})")
    assert abs(x - 60) < 0.1
    assert abs(y - 70) < 0.1
    
    print("\n✅ Coordinate mapping tests passed!")

if __name__ == "__main__":
    test_coordinate_mapping()
