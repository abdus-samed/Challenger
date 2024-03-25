// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__SIGNS__TRAITS_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__SIGNS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "team_interfaces/msg/detail/signs__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace team_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Signs & msg,
  std::ostream & out)
{
  out << "{";
  // member: sign_detected
  {
    out << "sign_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.sign_detected, out);
    out << ", ";
  }

  // member: overtaking_allowed
  {
    out << "overtaking_allowed: ";
    rosidl_generator_traits::value_to_yaml(msg.overtaking_allowed, out);
    out << ", ";
  }

  // member: overtaking_forbidden
  {
    out << "overtaking_forbidden: ";
    rosidl_generator_traits::value_to_yaml(msg.overtaking_forbidden, out);
    out << ", ";
  }

  // member: cross_parking
  {
    out << "cross_parking: ";
    rosidl_generator_traits::value_to_yaml(msg.cross_parking, out);
    out << ", ";
  }

  // member: parallel_parking
  {
    out << "parallel_parking: ";
    rosidl_generator_traits::value_to_yaml(msg.parallel_parking, out);
    out << ", ";
  }

  // member: pit_in
  {
    out << "pit_in: ";
    rosidl_generator_traits::value_to_yaml(msg.pit_in, out);
    out << ", ";
  }

  // member: pit_out
  {
    out << "pit_out: ";
    rosidl_generator_traits::value_to_yaml(msg.pit_out, out);
    out << ", ";
  }

  // member: sign_height
  {
    out << "sign_height: ";
    rosidl_generator_traits::value_to_yaml(msg.sign_height, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Signs & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: sign_detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sign_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.sign_detected, out);
    out << "\n";
  }

  // member: overtaking_allowed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "overtaking_allowed: ";
    rosidl_generator_traits::value_to_yaml(msg.overtaking_allowed, out);
    out << "\n";
  }

  // member: overtaking_forbidden
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "overtaking_forbidden: ";
    rosidl_generator_traits::value_to_yaml(msg.overtaking_forbidden, out);
    out << "\n";
  }

  // member: cross_parking
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cross_parking: ";
    rosidl_generator_traits::value_to_yaml(msg.cross_parking, out);
    out << "\n";
  }

  // member: parallel_parking
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "parallel_parking: ";
    rosidl_generator_traits::value_to_yaml(msg.parallel_parking, out);
    out << "\n";
  }

  // member: pit_in
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pit_in: ";
    rosidl_generator_traits::value_to_yaml(msg.pit_in, out);
    out << "\n";
  }

  // member: pit_out
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pit_out: ";
    rosidl_generator_traits::value_to_yaml(msg.pit_out, out);
    out << "\n";
  }

  // member: sign_height
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sign_height: ";
    rosidl_generator_traits::value_to_yaml(msg.sign_height, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Signs & msg, bool use_flow_style = false)
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
  const team_interfaces::msg::Signs & msg,
  std::ostream & out, size_t indentation = 0)
{
  team_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use team_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const team_interfaces::msg::Signs & msg)
{
  return team_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<team_interfaces::msg::Signs>()
{
  return "team_interfaces::msg::Signs";
}

template<>
inline const char * name<team_interfaces::msg::Signs>()
{
  return "team_interfaces/msg/Signs";
}

template<>
struct has_fixed_size<team_interfaces::msg::Signs>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<team_interfaces::msg::Signs>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<team_interfaces::msg::Signs>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TEAM_INTERFACES__MSG__DETAIL__SIGNS__TRAITS_HPP_
