// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/signs__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "team_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "team_interfaces/msg/detail/signs__struct.h"
#include "team_interfaces/msg/detail/signs__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _Signs__ros_msg_type = team_interfaces__msg__Signs;

static bool _Signs__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Signs__ros_msg_type * ros_message = static_cast<const _Signs__ros_msg_type *>(untyped_ros_message);
  // Field name: sign_detected
  {
    cdr << (ros_message->sign_detected ? true : false);
  }

  // Field name: overtaking_allowed
  {
    cdr << (ros_message->overtaking_allowed ? true : false);
  }

  // Field name: overtaking_forbidden
  {
    cdr << (ros_message->overtaking_forbidden ? true : false);
  }

  // Field name: cross_parking
  {
    cdr << (ros_message->cross_parking ? true : false);
  }

  // Field name: parallel_parking
  {
    cdr << (ros_message->parallel_parking ? true : false);
  }

  // Field name: pit_in
  {
    cdr << (ros_message->pit_in ? true : false);
  }

  // Field name: pit_out
  {
    cdr << (ros_message->pit_out ? true : false);
  }

  // Field name: sign_height
  {
    cdr << ros_message->sign_height;
  }

  return true;
}

static bool _Signs__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Signs__ros_msg_type * ros_message = static_cast<_Signs__ros_msg_type *>(untyped_ros_message);
  // Field name: sign_detected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->sign_detected = tmp ? true : false;
  }

  // Field name: overtaking_allowed
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->overtaking_allowed = tmp ? true : false;
  }

  // Field name: overtaking_forbidden
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->overtaking_forbidden = tmp ? true : false;
  }

  // Field name: cross_parking
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->cross_parking = tmp ? true : false;
  }

  // Field name: parallel_parking
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->parallel_parking = tmp ? true : false;
  }

  // Field name: pit_in
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->pit_in = tmp ? true : false;
  }

  // Field name: pit_out
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->pit_out = tmp ? true : false;
  }

  // Field name: sign_height
  {
    cdr >> ros_message->sign_height;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_team_interfaces
size_t get_serialized_size_team_interfaces__msg__Signs(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Signs__ros_msg_type * ros_message = static_cast<const _Signs__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name sign_detected
  {
    size_t item_size = sizeof(ros_message->sign_detected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name overtaking_allowed
  {
    size_t item_size = sizeof(ros_message->overtaking_allowed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name overtaking_forbidden
  {
    size_t item_size = sizeof(ros_message->overtaking_forbidden);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name cross_parking
  {
    size_t item_size = sizeof(ros_message->cross_parking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name parallel_parking
  {
    size_t item_size = sizeof(ros_message->parallel_parking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name pit_in
  {
    size_t item_size = sizeof(ros_message->pit_in);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name pit_out
  {
    size_t item_size = sizeof(ros_message->pit_out);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name sign_height
  {
    size_t item_size = sizeof(ros_message->sign_height);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Signs__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_team_interfaces__msg__Signs(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_team_interfaces
size_t max_serialized_size_team_interfaces__msg__Signs(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: sign_detected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: overtaking_allowed
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: overtaking_forbidden
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: cross_parking
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: parallel_parking
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: pit_in
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: pit_out
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: sign_height
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = team_interfaces__msg__Signs;
    is_plain =
      (
      offsetof(DataType, sign_height) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _Signs__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_team_interfaces__msg__Signs(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_Signs = {
  "team_interfaces::msg",
  "Signs",
  _Signs__cdr_serialize,
  _Signs__cdr_deserialize,
  _Signs__get_serialized_size,
  _Signs__max_serialized_size
};

static rosidl_message_type_support_t _Signs__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Signs,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, team_interfaces, msg, Signs)() {
  return &_Signs__type_support;
}

#if defined(__cplusplus)
}
#endif
