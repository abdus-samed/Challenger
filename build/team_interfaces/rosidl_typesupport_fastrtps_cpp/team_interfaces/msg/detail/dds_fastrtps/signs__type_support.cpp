// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/signs__rosidl_typesupport_fastrtps_cpp.hpp"
#include "team_interfaces/msg/detail/signs__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace team_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team_interfaces
cdr_serialize(
  const team_interfaces::msg::Signs & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: sign_detected
  cdr << (ros_message.sign_detected ? true : false);
  // Member: overtaking_allowed
  cdr << (ros_message.overtaking_allowed ? true : false);
  // Member: overtaking_forbidden
  cdr << (ros_message.overtaking_forbidden ? true : false);
  // Member: cross_parking
  cdr << (ros_message.cross_parking ? true : false);
  // Member: parallel_parking
  cdr << (ros_message.parallel_parking ? true : false);
  // Member: pit_in
  cdr << (ros_message.pit_in ? true : false);
  // Member: pit_out
  cdr << (ros_message.pit_out ? true : false);
  // Member: sign_height
  cdr << ros_message.sign_height;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  team_interfaces::msg::Signs & ros_message)
{
  // Member: sign_detected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.sign_detected = tmp ? true : false;
  }

  // Member: overtaking_allowed
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.overtaking_allowed = tmp ? true : false;
  }

  // Member: overtaking_forbidden
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.overtaking_forbidden = tmp ? true : false;
  }

  // Member: cross_parking
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.cross_parking = tmp ? true : false;
  }

  // Member: parallel_parking
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.parallel_parking = tmp ? true : false;
  }

  // Member: pit_in
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.pit_in = tmp ? true : false;
  }

  // Member: pit_out
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.pit_out = tmp ? true : false;
  }

  // Member: sign_height
  cdr >> ros_message.sign_height;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team_interfaces
get_serialized_size(
  const team_interfaces::msg::Signs & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: sign_detected
  {
    size_t item_size = sizeof(ros_message.sign_detected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: overtaking_allowed
  {
    size_t item_size = sizeof(ros_message.overtaking_allowed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: overtaking_forbidden
  {
    size_t item_size = sizeof(ros_message.overtaking_forbidden);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: cross_parking
  {
    size_t item_size = sizeof(ros_message.cross_parking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: parallel_parking
  {
    size_t item_size = sizeof(ros_message.parallel_parking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: pit_in
  {
    size_t item_size = sizeof(ros_message.pit_in);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: pit_out
  {
    size_t item_size = sizeof(ros_message.pit_out);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: sign_height
  {
    size_t item_size = sizeof(ros_message.sign_height);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_team_interfaces
max_serialized_size_Signs(
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


  // Member: sign_detected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: overtaking_allowed
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: overtaking_forbidden
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: cross_parking
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: parallel_parking
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: pit_in
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: pit_out
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: sign_height
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
    using DataType = team_interfaces::msg::Signs;
    is_plain =
      (
      offsetof(DataType, sign_height) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _Signs__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const team_interfaces::msg::Signs *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _Signs__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<team_interfaces::msg::Signs *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _Signs__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const team_interfaces::msg::Signs *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _Signs__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_Signs(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _Signs__callbacks = {
  "team_interfaces::msg",
  "Signs",
  _Signs__cdr_serialize,
  _Signs__cdr_deserialize,
  _Signs__get_serialized_size,
  _Signs__max_serialized_size
};

static rosidl_message_type_support_t _Signs__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_Signs__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace team_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_team_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<team_interfaces::msg::Signs>()
{
  return &team_interfaces::msg::typesupport_fastrtps_cpp::_Signs__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, team_interfaces, msg, Signs)() {
  return &team_interfaces::msg::typesupport_fastrtps_cpp::_Signs__handle;
}

#ifdef __cplusplus
}
#endif
