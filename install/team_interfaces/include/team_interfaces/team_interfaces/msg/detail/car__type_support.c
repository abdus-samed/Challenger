// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from team_interfaces:msg/Car.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "team_interfaces/msg/detail/car__rosidl_typesupport_introspection_c.h"
#include "team_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "team_interfaces/msg/detail/car__functions.h"
#include "team_interfaces/msg/detail/car__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  team_interfaces__msg__Car__init(message_memory);
}

void team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_fini_function(void * message_memory)
{
  team_interfaces__msg__Car__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_member_array[2] = {
  {
    "detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Car, detected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "distance",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(team_interfaces__msg__Car, distance),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_members = {
  "team_interfaces__msg",  // message namespace
  "Car",  // message name
  2,  // number of fields
  sizeof(team_interfaces__msg__Car),
  team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_member_array,  // message members
  team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_init_function,  // function to initialize message memory (memory has to be allocated)
  team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_type_support_handle = {
  0,
  &team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_team_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team_interfaces, msg, Car)() {
  if (!team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_type_support_handle.typesupport_identifier) {
    team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &team_interfaces__msg__Car__rosidl_typesupport_introspection_c__Car_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
