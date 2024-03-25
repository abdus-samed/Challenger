// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from team_interfaces:msg/Car.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__CAR__STRUCT_H_
#define TEAM_INTERFACES__MSG__DETAIL__CAR__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Car in the package team_interfaces.
/**
  * Message to define car obstacle
 */
typedef struct team_interfaces__msg__Car
{
  bool detected;
  float distance;
} team_interfaces__msg__Car;

// Struct for a sequence of team_interfaces__msg__Car.
typedef struct team_interfaces__msg__Car__Sequence
{
  team_interfaces__msg__Car * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} team_interfaces__msg__Car__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEAM_INTERFACES__MSG__DETAIL__CAR__STRUCT_H_
