from math import atan2


def timestamp_to_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return atan2(siny_cosp, cosy_cosp)
