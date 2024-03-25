// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team_interfaces:msg/Emergency.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__STRUCT_H_
#define TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Emergency in the package team_interfaces.
/**
  * Message for emergency stop
 */
typedef struct team_interfaces__msg__Emergency
{
  bool emergency_stop;
  bool phase_change;
} team_interfaces__msg__Emergency;

// Struct for a sequence of team_interfaces__msg__Emergency.
typedef struct team_interfaces__msg__Emergency__Sequence
{
  team_interfaces__msg__Emergency * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team_interfaces__msg__Emergency__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__STRUCT_H_
