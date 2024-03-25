// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__SIGNS__BUILDER_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__SIGNS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team_interfaces/msg/detail/signs__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team_interfaces
{

namespace msg
{

namespace builder
{

class Init_Signs_sign_height
{
public:
  explicit Init_Signs_sign_height(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  ::team_interfaces::msg::Signs sign_height(::team_interfaces::msg::Signs::_sign_height_type arg)
  {
    msg_.sign_height = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_pit_out
{
public:
  explicit Init_Signs_pit_out(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  Init_Signs_sign_height pit_out(::team_interfaces::msg::Signs::_pit_out_type arg)
  {
    msg_.pit_out = std::move(arg);
    return Init_Signs_sign_height(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_pit_in
{
public:
  explicit Init_Signs_pit_in(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  Init_Signs_pit_out pit_in(::team_interfaces::msg::Signs::_pit_in_type arg)
  {
    msg_.pit_in = std::move(arg);
    return Init_Signs_pit_out(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_parallel_parking
{
public:
  explicit Init_Signs_parallel_parking(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  Init_Signs_pit_in parallel_parking(::team_interfaces::msg::Signs::_parallel_parking_type arg)
  {
    msg_.parallel_parking = std::move(arg);
    return Init_Signs_pit_in(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_cross_parking
{
public:
  explicit Init_Signs_cross_parking(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  Init_Signs_parallel_parking cross_parking(::team_interfaces::msg::Signs::_cross_parking_type arg)
  {
    msg_.cross_parking = std::move(arg);
    return Init_Signs_parallel_parking(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_overtaking_forbidden
{
public:
  explicit Init_Signs_overtaking_forbidden(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  Init_Signs_cross_parking overtaking_forbidden(::team_interfaces::msg::Signs::_overtaking_forbidden_type arg)
  {
    msg_.overtaking_forbidden = std::move(arg);
    return Init_Signs_cross_parking(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_overtaking_allowed
{
public:
  explicit Init_Signs_overtaking_allowed(::team_interfaces::msg::Signs & msg)
  : msg_(msg)
  {}
  Init_Signs_overtaking_forbidden overtaking_allowed(::team_interfaces::msg::Signs::_overtaking_allowed_type arg)
  {
    msg_.overtaking_allowed = std::move(arg);
    return Init_Signs_overtaking_forbidden(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

class Init_Signs_sign_detected
{
public:
  Init_Signs_sign_detected()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Signs_overtaking_allowed sign_detected(::team_interfaces::msg::Signs::_sign_detected_type arg)
  {
    msg_.sign_detected = std::move(arg);
    return Init_Signs_overtaking_allowed(msg_);
  }

private:
  ::team_interfaces::msg::Signs msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::team_interfaces::msg::Signs>()
{
  return team_interfaces::msg::builder::Init_Signs_sign_detected();
}

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__SIGNS__BUILDER_HPP_
