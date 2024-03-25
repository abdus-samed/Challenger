// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "team_interfaces/msg/detail/lane__rosidl_typesupport_introspection_c.h"
#include "team_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "team_interfaces/msg/detail/lane__functions.h"
#include "team_interfaces/msg/detail/lane__struct.h"


// Include directives for member types
// Member `right_lane_x_points`
// Member `right_lane_y_points`
// Member `left_lane_x_points`
// Member `left_lane_y_points`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  team_interfaces__msg__Lane__init(message_memory);
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_fini_function(void * message_memory)
{
  team_interfaces__msg__Lane__fini(message_memory);
}

size_t team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__right_lane_x_points(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__right_lane_x_points(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__right_lane_x_points(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__right_lane_x_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__right_lane_x_points(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__right_lane_x_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__right_lane_x_points(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__right_lane_x_points(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__right_lane_y_points(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__right_lane_y_points(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__right_lane_y_points(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__right_lane_y_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__right_lane_y_points(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__right_lane_y_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__right_lane_y_points(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__right_lane_y_points(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__left_lane_x_points(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__left_lane_x_points(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__left_lane_x_points(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__left_lane_x_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__left_lane_x_points(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__left_lane_x_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__left_lane_x_points(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__left_lane_x_points(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__left_lane_y_points(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__left_lane_y_points(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__left_lane_y_points(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__left_lane_y_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__left_lane_y_points(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__left_lane_y_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__left_lane_y_points(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__left_lane_y_points(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_member_array[9] = {
  {
    "right_lane_detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, right_lane_detected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_lane_detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, left_lane_detected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "dashed_line_detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, dashed_line_detected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "center_offset",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, center_offset),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "heading_angle",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, heading_angle),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_lane_x_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, right_lane_x_points),  // bytes offset in struct
    NULL,  // default value
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__right_lane_x_points,  // size() function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__right_lane_x_points,  // get_const(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__right_lane_x_points,  // get(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__right_lane_x_points,  // fetch(index, &value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__right_lane_x_points,  // assign(index, value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__right_lane_x_points  // resize(index) function pointer
  },
  {
    "right_lane_y_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, right_lane_y_points),  // bytes offset in struct
    NULL,  // default value
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__right_lane_y_points,  // size() function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__right_lane_y_points,  // get_const(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__right_lane_y_points,  // get(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__right_lane_y_points,  // fetch(index, &value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__right_lane_y_points,  // assign(index, value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__right_lane_y_points  // resize(index) function pointer
  },
  {
    "left_lane_x_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, left_lane_x_points),  // bytes offset in struct
    NULL,  // default value
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__left_lane_x_points,  // size() function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__left_lane_x_points,  // get_const(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__left_lane_x_points,  // get(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__left_lane_x_points,  // fetch(index, &value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__left_lane_x_points,  // assign(index, value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__left_lane_x_points  // resize(index) function pointer
  },
  {
    "left_lane_y_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Lane, left_lane_y_points),  // bytes offset in struct
    NULL,  // default value
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__size_function__Lane__left_lane_y_points,  // size() function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_const_function__Lane__left_lane_y_points,  // get_const(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__get_function__Lane__left_lane_y_points,  // get(index) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__fetch_function__Lane__left_lane_y_points,  // fetch(index, &value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__assign_function__Lane__left_lane_y_points,  // assign(index, value) function pointer
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__resize_function__Lane__left_lane_y_points  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_members = {
  "team_interfaces__msg",  // message namespace
  "Lane",  // message name
  9,  // number of fields
  sizeof(team_interfaces__msg__Lane),
  team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_member_array,  // message members
  team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_init_function,  // function to initialize message memory (memory has to be allocated)
  team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_type_support_handle = {
  0,
  &team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_team_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team_interfaces, msg, Lane)() {
  if (!team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_type_support_handle.typesupport_identifier) {
    team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &team_interfaces__msg__Lane__rosidl_typesupport_introspection_c__Lane_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
