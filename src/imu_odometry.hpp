#pragma once

// High-rate IMU odometry using midpoint integration and correction-time replay.
// Gravity and acceleration scaling follow the FAST-LIO frontend conventions.

#include <deque>
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>

class ImuOdometry
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct State
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d bias_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d bias_gyr = Eigen::Vector3d::Zero();
        Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.809);
    };

    void reset()
    {
        ready_ = false;
        buffer_.clear();
    }

    bool addImu(const sensor_msgs::Imu &imu)
    {
        Measurement sample;
        sample.time = imu.header.stamp.toSec();
        sample.acceleration << imu.linear_acceleration.x,
                               imu.linear_acceleration.y,
                               imu.linear_acceleration.z;
        sample.angular_velocity << imu.angular_velocity.x,
                                   imu.angular_velocity.y,
                                   imu.angular_velocity.z;
        if (!buffer_.empty() && sample.time <= buffer_.back().time) return false;
        buffer_.push_back(sample);
        return ready_ && propagate(sample);
    }

    // Re-anchor at the LiDAR correction time, then catch up using received IMUs.
    // Replay updates the prediction only; it never republishes old timestamps.
    void correct(double time, const State &state, double acceleration_scale)
    {
        ready_ = false;
        while (buffer_.size() > 1 && buffer_[1].time <= time) buffer_.pop_front();
        if (buffer_.empty() || buffer_.front().time > time) return;

        state_ = state;
        acceleration_scale_ = acceleration_scale;
        previous_ = buffer_.front();
        // The correction usually falls between two IMU samples.
        if (buffer_.size() > 1)
        {
            const Measurement &next = buffer_[1];
            const double fraction = (time - previous_.time) / (next.time - previous_.time);
            previous_.acceleration += fraction * (next.acceleration - previous_.acceleration);
            previous_.angular_velocity += fraction * (next.angular_velocity - previous_.angular_velocity);
        }
        previous_.time = time;
        ready_ = true;
        for (const Measurement &sample : buffer_)
        {
            if (sample.time > time) propagate(sample);
        }
    }

    nav_msgs::Odometry odometry() const
    {
        nav_msgs::Odometry message;
        message.header.stamp.fromSec(previous_.time);
        message.header.frame_id = "camera_init";
        message.child_frame_id = "body";
        message.pose.pose.position.x = state_.position.x();
        message.pose.pose.position.y = state_.position.y();
        message.pose.pose.position.z = state_.position.z();
        message.pose.pose.orientation.x = state_.rotation.x();
        message.pose.pose.orientation.y = state_.rotation.y();
        message.pose.pose.orientation.z = state_.rotation.z();
        message.pose.pose.orientation.w = state_.rotation.w();
        const Eigen::Vector3d velocity_body = state_.rotation.conjugate() * state_.velocity;
        const Eigen::Vector3d angular_velocity = previous_.angular_velocity - state_.bias_gyr;
        message.twist.twist.linear.x = velocity_body.x();
        message.twist.twist.linear.y = velocity_body.y();
        message.twist.twist.linear.z = velocity_body.z();
        message.twist.twist.angular.x = angular_velocity.x();
        message.twist.twist.angular.y = angular_velocity.y();
        message.twist.twist.angular.z = angular_velocity.z();
        return message;
    }

private:
    struct Measurement
    {
        double time = 0.0;
        Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
        Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    };

    bool propagate(const Measurement &sample)
    {
        const double dt = sample.time - previous_.time;
        if (dt <= 0.0) return false;
        const Eigen::Vector3d acceleration_start =
            state_.rotation * (previous_.acceleration * acceleration_scale_ - state_.bias_acc) + state_.gravity;
        const Eigen::Vector3d angular_velocity =
            0.5 * (previous_.angular_velocity + sample.angular_velocity) - state_.bias_gyr;
        const double angular_speed = angular_velocity.norm();
        if (angular_speed > 1e-12)
        {
            state_.rotation = (state_.rotation * Eigen::Quaterniond(
                Eigen::AngleAxisd(angular_speed * dt, angular_velocity / angular_speed))).normalized();
        }
        const Eigen::Vector3d acceleration_end =
            state_.rotation * (sample.acceleration * acceleration_scale_ - state_.bias_acc) + state_.gravity;
        const Eigen::Vector3d acceleration = 0.5 * (acceleration_start + acceleration_end);
        state_.position += state_.velocity * dt + 0.5 * acceleration * dt * dt;
        state_.velocity += acceleration * dt;
        previous_ = sample;
        return true;
    }

    bool ready_ = false;
    double acceleration_scale_ = 1.0;
    State state_;
    Measurement previous_;
    std::deque<Measurement> buffer_;
};
