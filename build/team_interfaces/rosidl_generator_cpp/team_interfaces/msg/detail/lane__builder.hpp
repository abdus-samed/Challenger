// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__LANE__BUILDER_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__LANE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team_interfaces/msg/detail/lane__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team_interfaces
{

namespace msg
{

namespace builder
{

class Init_Lane_left_lane_y_points
{
public:
  explicit Init_Lane_left_lane_y_points(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  ::team_interfaces::msg::Lane left_lane_y_points(::team_interfaces::msg::Lane::_left_lane_y_points_type arg)
  {
    msg_.left_lane_y_points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_left_lane_x_points
{
public:
  explicit Init_Lane_left_lane_x_points(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_left_lane_y_points left_lane_x_points(::team_interfaces::msg::Lane::_left_lane_x_points_type arg)
  {
    msg_.left_lane_x_points = std::move(arg);
    return Init_Lane_left_lane_y_points(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_right_lane_y_points
{
public:
  explicit Init_Lane_right_lane_y_points(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_left_lane_x_points right_lane_y_points(::team_interfaces::msg::Lane::_right_lane_y_points_type arg)
  {
    msg_.right_lane_y_points = std::move(arg);
    return Init_Lane_left_lane_x_points(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_right_lane_x_points
{
public:
  explicit Init_Lane_right_lane_x_points(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_right_lane_y_points right_lane_x_points(::team_interfaces::msg::Lane::_right_lane_x_points_type arg)
  {
    msg_.right_lane_x_points = std::move(arg);
    return Init_Lane_right_lane_y_points(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_heading_angle
{
public:
  explicit Init_Lane_heading_angle(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_right_lane_x_points heading_angle(::team_interfaces::msg::Lane::_heading_angle_type arg)
  {
    msg_.heading_angle = std::move(arg);
    return Init_Lane_right_lane_x_points(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_center_offset
{
public:
  explicit Init_Lane_center_offset(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_heading_angle center_offset(::team_interfaces::msg::Lane::_center_offset_type arg)
  {
    msg_.center_offset = std::move(arg);
    return Init_Lane_heading_angle(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_dashed_line_detected
{
public:
  explicit Init_Lane_dashed_line_detected(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_center_offset dashed_line_detected(::team_interfaces::msg::Lane::_dashed_line_detected_type arg)
  {
    msg_.dashed_line_detected = std::move(arg);
    return Init_Lane_center_offset(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_left_lane_detected
{
public:
  explicit Init_Lane_left_lane_detected(::team_interfaces::msg::Lane & msg)
  : msg_(msg)
  {}
  Init_Lane_dashed_line_detected left_lane_detected(::team_interfaces::msg::Lane::_left_lane_detected_type arg)
  {
    msg_.left_lane_detected = std::move(arg);
    return Init_Lane_dashed_line_detected(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

class Init_Lane_right_lane_detected
{
public:
  Init_Lane_right_lane_detected()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Lane_left_lane_detected right_lane_detected(::team_interfaces::msg::Lane::_right_lane_detected_type arg)
  {
    msg_.right_lane_detected = std::move(arg);
    return Init_Lane_left_lane_detected(msg_);
  }

private:
  ::team_interfaces::msg::Lane msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::team_interfaces::msg::Lane>()
{
  return team_interfaces::msg::builder::Init_Lane_right_lane_detected();
}

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__LANE__BUILDER_HPP_
