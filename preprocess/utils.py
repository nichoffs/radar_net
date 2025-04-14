

def timestamp_to_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
