// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team_interfaces:msg/Car.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__CAR__BUILDER_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__CAR__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team_interfaces/msg/detail/car__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team_interfaces
{

namespace msg
{

namespace builder
{

class Init_Car_distance
{
public:
  explicit Init_Car_distance(::team_interfaces::msg::Car & msg)
  : msg_(msg)
  {}
  ::team_interfaces::msg::Car distance(::team_interfaces::msg::Car::_distance_type arg)
  {
    msg_.distance = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team_interfaces::msg::Car msg_;
};

class Init_Car_detected
{
public:
  Init_Car_detected()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Car_distance detected(::team_interfaces::msg::Car::_detected_type arg)
  {
    msg_.detected = std::move(arg);
    return Init_Car_distance(msg_);
  }

private:
  ::team_interfaces::msg::Car msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::team_interfaces::msg::Car>()
{
  return team_interfaces::msg::builder::Init_Car_detected();
}

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__CAR__BUILDER_HPP_
