// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from team_interfaces:msg/Trafficlight.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__BUILDER_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "team_interfaces/msg/detail/trafficlight__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace team_interfaces
{

namespace msg
{

namespace builder
{

class Init_Trafficlight_traffic_light
{
public:
  Init_Trafficlight_traffic_light()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::team_interfaces::msg::Trafficlight traffic_light(::team_interfaces::msg::Trafficlight::_traffic_light_type arg)
  {
    msg_.traffic_light = std::move(arg);
    return std::move(msg_);
  }

private:
  ::team_interfaces::msg::Trafficlight msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::team_interfaces::msg::Trafficlight>()
{
  return team_interfaces::msg::builder::Init_Trafficlight_traffic_light();
}

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__BUILDER_HPP_
