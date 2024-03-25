// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/lane__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "team_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "team_interfaces/msg/detail/lane__struct.h"
#include "team_interfaces/msg/detail/lane__functions.h"
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

#include "rosidl_runtime_c/primitives_sequence.h"  // left_lane_x_points, left_lane_y_points, right_lane_x_points, right_lane_y_points
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // left_lane_x_points, left_lane_y_points, right_lane_x_points, right_lane_y_points

// forward declare type support functions


using _Lane__ros_msg_type = team_interfaces__msg__Lane;

static bool _Lane__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Lane__ros_msg_type * ros_message = static_cast<const _Lane__ros_msg_type *>(untyped_ros_message);
  // Field name: right_lane_detected
  {
    cdr << (ros_message->right_lane_detected ? true : false);
  }

  // Field name: left_lane_detected
  {
    cdr << (ros_message->left_lane_detected ? true : false);
  }

  // Field name: dashed_line_detected
  {
    cdr << (ros_message->dashed_line_detected ? true : false);
  }

  // Field name: center_offset
  {
    cdr << ros_message->center_offset;
  }

  // Field name: heading_angle
  {
    cdr << ros_message->heading_angle;
  }

  // Field name: right_lane_x_points
  {
    size_t size = ros_message->right_lane_x_points.size;
    auto array_ptr = ros_message->right_lane_x_points.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: right_lane_y_points
  {
    size_t size = ros_message->right_lane_y_points.size;
    auto array_ptr = ros_message->right_lane_y_points.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: left_lane_x_points
  {
    size_t size = ros_message->left_lane_x_points.size;
    auto array_ptr = ros_message->left_lane_x_points.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: left_lane_y_points
  {
    size_t size = ros_message->left_lane_y_points.size;
    auto array_ptr = ros_message->left_lane_y_points.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _Lane__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Lane__ros_msg_type * ros_message = static_cast<_Lane__ros_msg_type *>(untyped_ros_message);
  // Field name: right_lane_detected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->right_lane_detected = tmp ? true : false;
  }

  // Field name: left_lane_detected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->left_lane_detected = tmp ? true : false;
  }

  // Field name: dashed_line_detected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->dashed_line_detected = tmp ? true : false;
  }

  // Field name: center_offset
  {
    cdr >> ros_message->center_offset;
  }

  // Field name: heading_angle
  {
    cdr >> ros_message->heading_angle;
  }

  // Field name: right_lane_x_points
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->right_lane_x_points.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->right_lane_x_points);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->right_lane_x_points, size)) {
      fprintf(stderr, "failed to create array for field 'right_lane_x_points'");
      return false;
    }
    auto array_ptr = ros_message->right_lane_x_points.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: right_lane_y_points
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->right_lane_y_points.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->right_lane_y_points);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->right_lane_y_points, size)) {
      fprintf(stderr, "failed to create array for field 'right_lane_y_points'");
      return false;
    }
    auto array_ptr = ros_message->right_lane_y_points.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: left_lane_x_points
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->left_lane_x_points.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->left_lane_x_points);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->left_lane_x_points, size)) {
      fprintf(stderr, "failed to create array for field 'left_lane_x_points'");
      return false;
    }
    auto array_ptr = ros_message->left_lane_x_points.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: left_lane_y_points
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->left_lane_y_points.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->left_lane_y_points);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->left_lane_y_points, size)) {
      fprintf(stderr, "failed to create array for field 'left_lane_y_points'");
      return false;
    }
    auto array_ptr = ros_message->left_lane_y_points.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_team_interfaces
size_t get_serialized_size_team_interfaces__msg__Lane(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Lane__ros_msg_type * ros_message = static_cast<const _Lane__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name right_lane_detected
  {
    size_t item_size = sizeof(ros_message->right_lane_detected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name left_lane_detected
  {
    size_t item_size = sizeof(ros_message->left_lane_detected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name dashed_line_detected
  {
    size_t item_size = sizeof(ros_message->dashed_line_detected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name center_offset
  {
    size_t item_size = sizeof(ros_message->center_offset);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name heading_angle
  {
    size_t item_size = sizeof(ros_message->heading_angle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name right_lane_x_points
  {
    size_t array_size = ros_message->right_lane_x_points.size;
    auto array_ptr = ros_message->right_lane_x_points.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name right_lane_y_points
  {
    size_t array_size = ros_message->right_lane_y_points.size;
    auto array_ptr = ros_message->right_lane_y_points.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name left_lane_x_points
  {
    size_t array_size = ros_message->left_lane_x_points.size;
    auto array_ptr = ros_message->left_lane_x_points.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name left_lane_y_points
  {
    size_t array_size = ros_message->left_lane_y_points.size;
    auto array_ptr = ros_message->left_lane_y_points.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Lane__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_team_interfaces__msg__Lane(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_team_interfaces
size_t max_serialized_size_team_interfaces__msg__Lane(
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

  // member: right_lane_detected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: left_lane_detected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: dashed_line_detected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: center_offset
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: heading_angle
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: right_lane_x_points
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: right_lane_y_points
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: left_lane_x_points
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: left_lane_y_points
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = team_interfaces__msg__Lane;
    is_plain =
      (
      offsetof(DataType, left_lane_y_points) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _Lane__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_team_interfaces__msg__Lane(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_Lane = {
  "team_interfaces::msg",
  "Lane",
  _Lane__cdr_serialize,
  _Lane__cdr_deserialize,
  _Lane__get_serialized_size,
  _Lane__max_serialized_size
};

static rosidl_message_type_support_t _Lane__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Lane,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, team_interfaces, msg, Lane)() {
  return &_Lane__type_support;
}

#if defined(__cplusplus)
}
#endif
