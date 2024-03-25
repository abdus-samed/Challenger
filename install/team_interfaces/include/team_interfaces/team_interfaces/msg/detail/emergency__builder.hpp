// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team_interfaces:msg/Emergency.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__BUILDER_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team_interfaces/msg/detail/emergency__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team_interfaces
{

namespace msg
{

namespace builder
{

class Init_Emergency_phase_change
{
public:
  explicit Init_Emergency_phase_change(::team_interfaces::msg::Emergency & msg)
  : msg_(msg)
  {}
  ::team_interfaces::msg::Emergency phase_change(::team_interfaces::msg::Emergency::_phase_change_type arg)
  {
    msg_.phase_change = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team_interfaces::msg::Emergency msg_;
};

class Init_Emergency_emergency_stop
{
public:
  Init_Emergency_emergency_stop()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Emergency_phase_change emergency_stop(::team_interfaces::msg::Emergency::_emergency_stop_type arg)
  {
    msg_.emergency_stop = std::move(arg);
    return Init_Emergency_phase_change(msg_);
  }

private:
  ::team_interfaces::msg::Emergency msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::team_interfaces::msg::Emergency>()
{
  return team_interfaces::msg::builder::Init_Emergency_emergency_stop();
}

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__BUILDER_HPP_
