// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/lane__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `right_lane_x_points`
// Member `right_lane_y_points`
// Member `left_lane_x_points`
// Member `left_lane_y_points`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
team_interfaces__msg__Lane__init(team_interfaces__msg__Lane * msg)
{
  if (!msg) {
    return false;
  }
  // right_lane_detected
  // left_lane_detected
  // dashed_line_detected
  // center_offset
  // heading_angle
  // right_lane_x_points
  if (!rosidl_runtime_c__float__Sequence__init(&msg->right_lane_x_points, 0)) {
    team_interfaces__msg__Lane__fini(msg);
    return false;
  }
  // right_lane_y_points
  if (!rosidl_runtime_c__float__Sequence__init(&msg->right_lane_y_points, 0)) {
    team_interfaces__msg__Lane__fini(msg);
    return false;
  }
  // left_lane_x_points
  if (!rosidl_runtime_c__float__Sequence__init(&msg->left_lane_x_points, 0)) {
    team_interfaces__msg__Lane__fini(msg);
    return false;
  }
  // left_lane_y_points
  if (!rosidl_runtime_c__float__Sequence__init(&msg->left_lane_y_points, 0)) {
    team_interfaces__msg__Lane__fini(msg);
    return false;
  }
  return true;
}

void
team_interfaces__msg__Lane__fini(team_interfaces__msg__Lane * msg)
{
  if (!msg) {
    return;
  }
  // right_lane_detected
  // left_lane_detected
  // dashed_line_detected
  // center_offset
  // heading_angle
  // right_lane_x_points
  rosidl_runtime_c__float__Sequence__fini(&msg->right_lane_x_points);
  // right_lane_y_points
  rosidl_runtime_c__float__Sequence__fini(&msg->right_lane_y_points);
  // left_lane_x_points
  rosidl_runtime_c__float__Sequence__fini(&msg->left_lane_x_points);
  // left_lane_y_points
  rosidl_runtime_c__float__Sequence__fini(&msg->left_lane_y_points);
}

bool
team_interfaces__msg__Lane__are_equal(const team_interfaces__msg__Lane * lhs, const team_interfaces__msg__Lane * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // right_lane_detected
  if (lhs->right_lane_detected != rhs->right_lane_detected) {
    return false;
  }
  // left_lane_detected
  if (lhs->left_lane_detected != rhs->left_lane_detected) {
    return false;
  }
  // dashed_line_detected
  if (lhs->dashed_line_detected != rhs->dashed_line_detected) {
    return false;
  }
  // center_offset
  if (lhs->center_offset != rhs->center_offset) {
    return false;
  }
  // heading_angle
  if (lhs->heading_angle != rhs->heading_angle) {
    return false;
  }
  // right_lane_x_points
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->right_lane_x_points), &(rhs->right_lane_x_points)))
  {
    return false;
  }
  // right_lane_y_points
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->right_lane_y_points), &(rhs->right_lane_y_points)))
  {
    return false;
  }
  // left_lane_x_points
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->left_lane_x_points), &(rhs->left_lane_x_points)))
  {
    return false;
  }
  // left_lane_y_points
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->left_lane_y_points), &(rhs->left_lane_y_points)))
  {
    return false;
  }
  return true;
}

bool
team_interfaces__msg__Lane__copy(
  const team_interfaces__msg__Lane * input,
  team_interfaces__msg__Lane * output)
{
  if (!input || !output) {
    return false;
  }
  // right_lane_detected
  output->right_lane_detected = input->right_lane_detected;
  // left_lane_detected
  output->left_lane_detected = input->left_lane_detected;
  // dashed_line_detected
  output->dashed_line_detected = input->dashed_line_detected;
  // center_offset
  output->center_offset = input->center_offset;
  // heading_angle
  output->heading_angle = input->heading_angle;
  // right_lane_x_points
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->right_lane_x_points), &(output->right_lane_x_points)))
  {
    return false;
  }
  // right_lane_y_points
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->right_lane_y_points), &(output->right_lane_y_points)))
  {
    return false;
  }
  // left_lane_x_points
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->left_lane_x_points), &(output->left_lane_x_points)))
  {
    return false;
  }
  // left_lane_y_points
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->left_lane_y_points), &(output->left_lane_y_points)))
  {
    return false;
  }
  return true;
}

team_interfaces__msg__Lane *
team_interfaces__msg__Lane__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Lane * msg = (team_interfaces__msg__Lane *)allocator.allocate(sizeof(team_interfaces__msg__Lane), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team_interfaces__msg__Lane));
  bool success = team_interfaces__msg__Lane__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team_interfaces__msg__Lane__destroy(team_interfaces__msg__Lane * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team_interfaces__msg__Lane__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team_interfaces__msg__Lane__Sequence__init(team_interfaces__msg__Lane__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Lane * data = NULL;

  if (size) {
    data = (team_interfaces__msg__Lane *)allocator.zero_allocate(size, sizeof(team_interfaces__msg__Lane), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team_interfaces__msg__Lane__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team_interfaces__msg__Lane__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
team_interfaces__msg__Lane__Sequence__fini(team_interfaces__msg__Lane__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      team_interfaces__msg__Lane__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

team_interfaces__msg__Lane__Sequence *
team_interfaces__msg__Lane__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Lane__Sequence * array = (team_interfaces__msg__Lane__Sequence *)allocator.allocate(sizeof(team_interfaces__msg__Lane__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team_interfaces__msg__Lane__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team_interfaces__msg__Lane__Sequence__destroy(team_interfaces__msg__Lane__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team_interfaces__msg__Lane__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team_interfaces__msg__Lane__Sequence__are_equal(const team_interfaces__msg__Lane__Sequence * lhs, const team_interfaces__msg__Lane__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team_interfaces__msg__Lane__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team_interfaces__msg__Lane__Sequence__copy(
  const team_interfaces__msg__Lane__Sequence * input,
  team_interfaces__msg__Lane__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team_interfaces__msg__Lane);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team_interfaces__msg__Lane * data =
      (team_interfaces__msg__Lane *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team_interfaces__msg__Lane__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team_interfaces__msg__Lane__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team_interfaces__msg__Lane__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
