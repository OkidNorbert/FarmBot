# ik_solver.py
import math

def solve_2link_ik(x, y, l1, l2):
    """
    Analytical IK for planar 2-link arm.
    Returns (theta1_deg, theta2_deg) in degrees if solvable, else None.
    Angles are for servo direct mapping; you may need offsets.
    """
    # distance to target
    r = math.hypot(x, y)
    if r > (l1 + l2) or r < abs(l1 - l2):
        return None  # unreachable

    # law of cosines
    cos_q2 = (r*r - l1*l1 - l2*l2) / (2 * l1 * l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = math.acos(cos_q2)

    # angle between l1 and target vector
    k1 = l1 + l2 * cos_q2
    k2 = l2 * math.sin(q2)
    q1 = math.atan2(y, x) - math.atan2(k2, k1)

    theta1 = math.degrees(q1)
    theta2 = math.degrees(q2)
    # Depending on servo conventions you might need to use theta2_sign = -theta2 etc.
    return (theta1, theta2)

if __name__ == '__main__':
    # quick test
    print(solve_2link_ik(15, 10, 10, 10))
