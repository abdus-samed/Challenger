// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__SIGNS__STRUCT_H_
#define TEAM_INTERFACES__MSG__DETAIL__SIGNS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Signs in the package team_interfaces.
/**
  * Custom message to define signs
 */
typedef struct team_interfaces__msg__Signs
{
  bool sign_detected;
  bool overtaking_allowed;
  bool overtaking_forbidden;
  /// If false, parallel parking
  bool cross_parking;
  bool parallel_parking;
  bool pit_in;
  bool pit_out;
  float sign_height;
} team_interfaces__msg__Signs;

// Struct for a sequence of team_interfaces__msg__Signs.
typedef struct team_interfaces__msg__Signs__Sequence
{
  team_interfaces__msg__Signs * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team_interfaces__msg__Signs__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM_INTERFACES__MSG__DETAIL__SIGNS__STRUCT_H_
