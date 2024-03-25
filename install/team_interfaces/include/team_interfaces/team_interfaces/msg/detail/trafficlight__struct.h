// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team_interfaces:msg/Trafficlight.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__STRUCT_H_
#define TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Trafficlight in the package team_interfaces.
typedef struct team_interfaces__msg__Trafficlight
{
  bool traffic_light;
} team_interfaces__msg__Trafficlight;

// Struct for a sequence of team_interfaces__msg__Trafficlight.
typedef struct team_interfaces__msg__Trafficlight__Sequence
{
  team_interfaces__msg__Trafficlight * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team_interfaces__msg__Trafficlight__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__STRUCT_H_
