// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__LANE__STRUCT_H_
#define TEAM_INTERFACES__MSG__DETAIL__LANE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'right_lane_x_points'
// Member 'right_lane_y_points'
// Member 'left_lane_x_points'
// Member 'left_lane_y_points'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/Lane in the package team_interfaces.
/**
  * Message for detected lane lines (currently dashed_line_detected is a placeholder)
 */
typedef struct team_interfaces__msg__Lane
{
  /// Bool for detected lane
  bool right_lane_detected;
  bool left_lane_detected;
  bool dashed_line_detected;
  ///  Mean curvature from left and right lane lines (place holder)
  /// float32 curvature_radius
  ///  Offset to center point
  float center_offset;
  /// Heading angle
  float heading_angle;
  /// Array of detected lane line points
  rosidl_runtime_c__float__Sequence right_lane_x_points;
  rosidl_runtime_c__float__Sequence right_lane_y_points;
  rosidl_runtime_c__float__Sequence left_lane_x_points;
  rosidl_runtime_c__float__Sequence left_lane_y_points;
} team_interfaces__msg__Lane;

// Struct for a sequence of team_interfaces__msg__Lane.
typedef struct team_interfaces__msg__Lane__Sequence
{
  team_interfaces__msg__Lane * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team_interfaces__msg__Lane__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM_INTERFACES__MSG__DETAIL__LANE__STRUCT_H_
