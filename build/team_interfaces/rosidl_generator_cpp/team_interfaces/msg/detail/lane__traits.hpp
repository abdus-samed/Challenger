// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__LANE__TRAITS_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__LANE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "team_interfaces/msg/detail/lane__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace team_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Lane & msg,
  std::ostream & out)
{
  out << "{";
  // member: right_lane_detected
  {
    out << "right_lane_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.right_lane_detected, out);
    out << ", ";
  }

  // member: left_lane_detected
  {
    out << "left_lane_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.left_lane_detected, out);
    out << ", ";
  }

  // member: dashed_line_detected
  {
    out << "dashed_line_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.dashed_line_detected, out);
    out << ", ";
  }

  // member: center_offset
  {
    out << "center_offset: ";
    rosidl_generator_traits::value_to_yaml(msg.center_offset, out);
    out << ", ";
  }

  // member: heading_angle
  {
    out << "heading_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.heading_angle, out);
    out << ", ";
  }

  // member: right_lane_x_points
  {
    if (msg.right_lane_x_points.size() == 0) {
      out << "right_lane_x_points: []";
    } else {
      out << "right_lane_x_points: [";
      size_t pending_items = msg.right_lane_x_points.size();
      for (auto item : msg.right_lane_x_points) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: right_lane_y_points
  {
    if (msg.right_lane_y_points.size() == 0) {
      out << "right_lane_y_points: []";
    } else {
      out << "right_lane_y_points: [";
      size_t pending_items = msg.right_lane_y_points.size();
      for (auto item : msg.right_lane_y_points) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: left_lane_x_points
  {
    if (msg.left_lane_x_points.size() == 0) {
      out << "left_lane_x_points: []";
    } else {
      out << "left_lane_x_points: [";
      size_t pending_items = msg.left_lane_x_points.size();
      for (auto item : msg.left_lane_x_points) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: left_lane_y_points
  {
    if (msg.left_lane_y_points.size() == 0) {
      out << "left_lane_y_points: []";
    } else {
      out << "left_lane_y_points: [";
      size_t pending_items = msg.left_lane_y_points.size();
      for (auto item : msg.left_lane_y_points) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Lane & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: right_lane_detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_lane_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.right_lane_detected, out);
    out << "\n";
  }

  // member: left_lane_detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_lane_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.left_lane_detected, out);
    out << "\n";
  }

  // member: dashed_line_detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dashed_line_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.dashed_line_detected, out);
    out << "\n";
  }

  // member: center_offset
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "center_offset: ";
    rosidl_generator_traits::value_to_yaml(msg.center_offset, out);
    out << "\n";
  }

  // member: heading_angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "heading_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.heading_angle, out);
    out << "\n";
  }

  // member: right_lane_x_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_lane_x_points.size() == 0) {
      out << "right_lane_x_points: []\n";
    } else {
      out << "right_lane_x_points:\n";
      for (auto item : msg.right_lane_x_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: right_lane_y_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_lane_y_points.size() == 0) {
      out << "right_lane_y_points: []\n";
    } else {
      out << "right_lane_y_points:\n";
      for (auto item : msg.right_lane_y_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: left_lane_x_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_lane_x_points.size() == 0) {
      out << "left_lane_x_points: []\n";
    } else {
      out << "left_lane_x_points:\n";
      for (auto item : msg.left_lane_x_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: left_lane_y_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_lane_y_points.size() == 0) {
      out << "left_lane_y_points: []\n";
    } else {
      out << "left_lane_y_points:\n";
      for (auto item : msg.left_lane_y_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Lane & msg, bool use_flow_style = false)
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
  const team_interfaces::msg::Lane & msg,
  std::ostream & out, size_t indentation = 0)
{
  team_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use team_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const team_interfaces::msg::Lane & msg)
{
  return team_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<team_interfaces::msg::Lane>()
{
  return "team_interfaces::msg::Lane";
}

template<>
inline const char * name<team_interfaces::msg::Lane>()
{
  return "team_interfaces/msg/Lane";
}

template<>
struct has_fixed_size<team_interfaces::msg::Lane>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<team_interfaces::msg::Lane>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<team_interfaces::msg::Lane>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TEAM_INTERFACES__MSG__DETAIL__LANE__TRAITS_HPP_
