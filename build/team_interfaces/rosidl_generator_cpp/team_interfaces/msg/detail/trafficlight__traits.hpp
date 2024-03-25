// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from team_interfaces:msg/Trafficlight.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__TRAITS_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "team_interfaces/msg/detail/trafficlight__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace team_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Trafficlight & msg,
  std::ostream & out)
{
  out << "{";
  // member: traffic_light
  {
    out << "traffic_light: ";
    rosidl_generator_traits::value_to_yaml(msg.traffic_light, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Trafficlight & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: traffic_light
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "traffic_light: ";
    rosidl_generator_traits::value_to_yaml(msg.traffic_light, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Trafficlight & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace team_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use team_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const team_interfaces::msg::Trafficlight & msg,
  std::ostream & out, size_t indentation = 0)
{
  team_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use team_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const team_interfaces::msg::Trafficlight & msg)
{
  return team_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<team_interfaces::msg::Trafficlight>()
{
  return "team_interfaces::msg::Trafficlight";
}

template<>
inline const char * name<team_interfaces::msg::Trafficlight>()
{
  return "team_interfaces/msg/Trafficlight";
}

template<>
struct has_fixed_size<team_interfaces::msg::Trafficlight>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<team_interfaces::msg::Trafficlight>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<team_interfaces::msg::Trafficlight>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__TRAITS_HPP_
