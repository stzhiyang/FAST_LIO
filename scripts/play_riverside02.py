#!/usr/bin/env python3
"""Publish Riverside02 Ouster/XSENS files for the FAST-LIO frontend.

The XYZI files have no point timestamps. Assume column-major 1024 x 64
scans, a constant rotation period and a configurable CSV timestamp reference.
The generated `t` is an approximation, not a recovered sensor measurement.
"""

import bisect
import csv
import heapq
import math
from pathlib import Path
import time

import numpy as np
import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2, PointField


NSEC = 1_000_000_000
RINGS = 64
COLUMNS = 1024
POINT_COUNT = RINGS * COLUMNS
POINT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "ambient", "range"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "u1", "<u2", "<u4"],
    "offsets": [0, 4, 8, 12, 16, 20, 22, 24, 28],
    "itemsize": 32,
})
POINT_FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("t", 16, PointField.UINT32, 1),
    PointField("reflectivity", 20, PointField.UINT16, 1),
    PointField("ring", 22, PointField.UINT8, 1),
    PointField("ambient", 24, PointField.UINT16, 1),
    PointField("range", 28, PointField.UINT32, 1),
]


def ros_time(stamp):
    """Keep CSV nanoseconds as integers, including the Unix epoch part."""
    return rospy.Time(stamp // NSEC, stamp % NSEC)


def load_inputs(dataset):
    with (dataset / "ouster_front_stamp.csv").open() as stream:
        lidar_stamps = [int(row[0]) for row in csv.reader(stream) if row]
    imu_rows = []
    with (dataset / "xsens_imu.csv").open() as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue
            if len(row) != 17:
                raise ValueError("xsens_imu.csv line %d: expected 17 columns" % number)
            # stamp, quaternion(4), Euler(3), gyro(3), acceleration(3), magnetic(3).
            stamp = int(row[0])
            values = tuple(float(value) for value in row[8:14])
            if not all(math.isfinite(value) for value in values):
                raise ValueError("xsens_imu.csv line %d: non-finite IMU value" % number)
            imu_rows.append((stamp, values))
    for name, stamps in [("Ouster", lidar_stamps), ("IMU", [row[0] for row in imu_rows])]:
        if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError("%s timestamps must be nonempty and strictly increasing" % name)
    if not (dataset / "Ouster").is_dir():
        raise ValueError("missing extracted directory: %s" % (dataset / "Ouster"))
    return lidar_stamps, imu_rows


def cloud_message(dataset, file_stamp, start_stamp, period_ns):
    filename = dataset / "Ouster" / (str(file_stamp) + ".bin")
    if filename.stat().st_size != POINT_COUNT * 4 * 4:
        raise ValueError("%s: expected 1024 x 64 float32 XYZI points" % filename)
    raw = np.fromfile(filename, dtype="<f4").reshape(POINT_COUNT, 4)
    indices = np.arange(POINT_COUNT, dtype=np.int64)
    ranges = np.linalg.norm(raw[:, :3], axis=1)
    valid = np.isfinite(raw).all(axis=1) & (ranges > 0.01)
    if not np.any(valid):
        raise ValueError("%s: no valid points" % filename)
    packed = np.zeros(int(valid.sum()), dtype=POINT_DTYPE)
    for column, name in enumerate(("x", "y", "z", "intensity")):
        packed[name] = raw[valid, column]
    # Assign ring/time before filtering so zero returns do not compress scan time.
    packed["ring"] = indices[valid] % RINGS
    packed["t"] = (indices[valid] // RINGS) * period_ns // COLUMNS
    packed["range"] = np.minimum(ranges[valid] * 1000.0, 2**32 - 1).astype("<u4")
    message = PointCloud2()
    message.header.stamp = ros_time(start_stamp)
    message.header.frame_id = "ouster"
    message.height = 1
    message.width = len(packed)
    message.fields = POINT_FIELDS
    message.is_bigendian = False
    message.point_step = POINT_DTYPE.itemsize
    message.row_step = message.width * message.point_step
    message.is_dense = True
    message.data = packed.tobytes()
    return message


def imu_message(stamp, values):
    message = Imu()
    message.header.stamp = ros_time(stamp)
    message.header.frame_id = "imu"
    # Only raw gyro/acceleration are used; the CSV AHRS attitude is not fused.
    message.orientation.w = 1.0
    message.orientation_covariance[0] = -1.0
    message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = values[:3]
    message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = values[3:]
    return message


def wait_until(deadline):
    # Wall time avoids waiting for the simulated clock that this node supplies.
    while not rospy.is_shutdown():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 0.05))
    return False


def play():
    dataset = Path(rospy.get_param("~dataset")).expanduser()
    rate = float(rospy.get_param("~rate", 1.0))
    start = float(rospy.get_param("~start", 0.0))
    duration = float(rospy.get_param("~duration", 0.0))
    delay = float(rospy.get_param("~delay", 2.0))
    period = float(rospy.get_param("~scan_period", 0.1))
    reference = rospy.get_param("~stamp_reference", "start")
    if not all(math.isfinite(v) for v in (rate, start, duration, delay, period)):
        raise ValueError("playback parameters must be finite")
    if rate <= 0 or min(start, duration, delay) < 0 or not 0 < period < 1:
        raise ValueError("rate > 0, start/duration/delay >= 0 and 0 < scan_period < 1 required")
    if reference not in ("start", "end"):
        raise ValueError("stamp_reference must be start or end")
    period_ns = round(period * NSEC)
    last_offset = (COLUMNS - 1) * period_ns // COLUMNS
    shift = last_offset if reference == "end" else 0
    lidar_stamps, imu_rows = load_inputs(dataset)
    lower = lidar_stamps[0] + round(start * NSEC)
    upper = lower + round(duration * NSEC) if duration else lidar_stamps[-1] + 1
    selected = [stamp for stamp in lidar_stamps if lower <= stamp < upper]
    # Do not extrapolate outside the recorded IMU interval at either boundary.
    usable = [stamp for stamp in selected
              if imu_rows[0][0] <= stamp - shift
              and stamp - shift + last_offset <= imu_rows[-1][0]]
    if not usable:
        raise ValueError("no selected lidar scans have full IMU coverage")
    if len(usable) != len(selected):
        rospy.logwarn("Skipped %d boundary scans without full IMU coverage", len(selected) - len(usable))
    imu_stamps = [row[0] for row in imu_rows]
    # Supply an IMU sample preceding the first scan and one following the last.
    first = max(0, bisect.bisect_right(imu_stamps, usable[0] - shift) - 1)
    last = min(len(imu_rows), bisect.bisect_left(imu_stamps, usable[-1] - shift + last_offset) + 1)
    imu_events = ((stamp, 0, values) for stamp, values in imu_rows[first:last])
    lidar_events = ((stamp - shift + last_offset, 1, stamp) for stamp in usable)
    events = heapq.merge(imu_events, lidar_events, key=lambda event: (event[0], event[1]))
    imu_pub = rospy.Publisher("/imu/data_raw", Imu, queue_size=200)
    lidar_pub = rospy.Publisher("/os1_points", PointCloud2, queue_size=10)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10, latch=True)
    rospy.logwarn("Point times are estimated from column order and %.3f s rotation; CSV reference assumed %s",
                  period, reference)
    rospy.loginfo("Waiting for frontend subscribers on /imu/data_raw and /os1_points")
    while not rospy.is_shutdown() and (not imu_pub.get_num_connections() or not lidar_pub.get_num_connections()):
        time.sleep(0.1)
    if not wait_until(time.monotonic() + delay):
        return
    rospy.loginfo("Playback started: %d lidar scans, rate %.2fx", len(usable), rate)
    base_stamp = imu_rows[first][0]
    wall_start = time.monotonic()
    imu_count = lidar_count = 0
    for stamp, kind, payload in events:
        if rospy.is_shutdown():
            return
        message = (imu_message(stamp, payload) if kind == 0
                   else cloud_message(dataset, payload, payload - shift, period_ns))
        if not wait_until(wall_start + (stamp - base_stamp) / NSEC / rate):
            return
        clock_pub.publish(Clock(clock=ros_time(stamp)))
        if kind == 0:
            imu_pub.publish(message)
            imu_count += 1
        else:
            lidar_pub.publish(message)
            lidar_count += 1
            if lidar_count == 1 or lidar_count % 50 == 0:
                rospy.loginfo("Playback: %d/%d lidar scans, %.1f s of data published",
                              lidar_count, len(usable), (stamp - base_stamp) / NSEC)
    rospy.loginfo("Playback finished: %d lidar scans, %d IMU samples; clock held, frontend/backend may finish and save",
                  lidar_count, imu_count)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("riverside02_player")
    try:
        play()
    except rospy.ROSInterruptException:
        pass
    except (OSError, ValueError) as error:
        rospy.logfatal("Riverside02 input error: %s", error)
        raise SystemExit(1)
