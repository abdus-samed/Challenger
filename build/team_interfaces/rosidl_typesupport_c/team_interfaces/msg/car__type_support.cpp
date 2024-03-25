// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from team_interfaces:msg/Car.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "team_interfaces/msg/detail/car__struct.h"
#include "team_interfaces/msg/detail/car__type_support.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace team_interfaces
{

namespace msg
{

namespace rosidl_typesupport_c
{

typedef struct _Car_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Car_type_support_ids_t;

static const _Car_type_support_ids_t _Car_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Car_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Car_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Car_type_support_symbol_names_t _Car_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, team_interfaces, msg, Car)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, team_interfaces, msg, Car)),
  }
};

typedef struct _Car_type_support_data_t
{
  void * data[2];
} _Car_type_support_data_t;

static _Car_type_support_data_t _Car_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Car_message_typesupport_map = {
  2,
  "team_interfaces",
  &_Car_message_typesupport_ids.typesupport_identifier[0],
  &_Car_message_typesupport_symbol_names.symbol_name[0],
  &_Car_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Car_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Car_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace msg

}  // namespace team_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, team_interfaces, msg, Car)() {
  return &::team_interfaces::msg::rosidl_typesupport_c::Car_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
