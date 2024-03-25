// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/signs__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
team_interfaces__msg__Signs__init(team_interfaces__msg__Signs * msg)
{
  if (!msg) {
    return false;
  }
  // sign_detected
  // overtaking_allowed
  // overtaking_forbidden
  // cross_parking
  // parallel_parking
  // pit_in
  // pit_out
  // sign_height
  return true;
}

void
team_interfaces__msg__Signs__fini(team_interfaces__msg__Signs * msg)
{
  if (!msg) {
    return;
  }
  // sign_detected
  // overtaking_allowed
  // overtaking_forbidden
  // cross_parking
  // parallel_parking
  // pit_in
  // pit_out
  // sign_height
}

bool
team_interfaces__msg__Signs__are_equal(const team_interfaces__msg__Signs * lhs, const team_interfaces__msg__Signs * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // sign_detected
  if (lhs->sign_detected != rhs->sign_detected) {
    return false;
  }
  // overtaking_allowed
  if (lhs->overtaking_allowed != rhs->overtaking_allowed) {
    return false;
  }
  // overtaking_forbidden
  if (lhs->overtaking_forbidden != rhs->overtaking_forbidden) {
    return false;
  }
  // cross_parking
  if (lhs->cross_parking != rhs->cross_parking) {
    return false;
  }
  // parallel_parking
  if (lhs->parallel_parking != rhs->parallel_parking) {
    return false;
  }
  // pit_in
  if (lhs->pit_in != rhs->pit_in) {
    return false;
  }
  // pit_out
  if (lhs->pit_out != rhs->pit_out) {
    return false;
  }
  // sign_height
  if (lhs->sign_height != rhs->sign_height) {
    return false;
  }
  return true;
}

bool
team_interfaces__msg__Signs__copy(
  const team_interfaces__msg__Signs * input,
  team_interfaces__msg__Signs * output)
{
  if (!input || !output) {
    return false;
  }
  // sign_detected
  output->sign_detected = input->sign_detected;
  // overtaking_allowed
  output->overtaking_allowed = input->overtaking_allowed;
  // overtaking_forbidden
  output->overtaking_forbidden = input->overtaking_forbidden;
  // cross_parking
  output->cross_parking = input->cross_parking;
  // parallel_parking
  output->parallel_parking = input->parallel_parking;
  // pit_in
  output->pit_in = input->pit_in;
  // pit_out
  output->pit_out = input->pit_out;
  // sign_height
  output->sign_height = input->sign_height;
  return true;
}

team_interfaces__msg__Signs *
team_interfaces__msg__Signs__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Signs * msg = (team_interfaces__msg__Signs *)allocator.allocate(sizeof(team_interfaces__msg__Signs), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team_interfaces__msg__Signs));
  bool success = team_interfaces__msg__Signs__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team_interfaces__msg__Signs__destroy(team_interfaces__msg__Signs * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team_interfaces__msg__Signs__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team_interfaces__msg__Signs__Sequence__init(team_interfaces__msg__Signs__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Signs * data = NULL;

  if (size) {
    data = (team_interfaces__msg__Signs *)allocator.zero_allocate(size, sizeof(team_interfaces__msg__Signs), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team_interfaces__msg__Signs__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team_interfaces__msg__Signs__fini(&data[i - 1]);
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
team_interfaces__msg__Signs__Sequence__fini(team_interfaces__msg__Signs__Sequence * array)
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
      team_interfaces__msg__Signs__fini(&array->data[i]);
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

team_interfaces__msg__Signs__Sequence *
team_interfaces__msg__Signs__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Signs__Sequence * array = (team_interfaces__msg__Signs__Sequence *)allocator.allocate(sizeof(team_interfaces__msg__Signs__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team_interfaces__msg__Signs__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team_interfaces__msg__Signs__Sequence__destroy(team_interfaces__msg__Signs__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team_interfaces__msg__Signs__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team_interfaces__msg__Signs__Sequence__are_equal(const team_interfaces__msg__Signs__Sequence * lhs, const team_interfaces__msg__Signs__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team_interfaces__msg__Signs__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team_interfaces__msg__Signs__Sequence__copy(
  const team_interfaces__msg__Signs__Sequence * input,
  team_interfaces__msg__Signs__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team_interfaces__msg__Signs);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team_interfaces__msg__Signs * data =
      (team_interfaces__msg__Signs *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team_interfaces__msg__Signs__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team_interfaces__msg__Signs__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team_interfaces__msg__Signs__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
