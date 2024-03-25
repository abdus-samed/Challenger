// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from team_interfaces:msg/Emergency.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/emergency__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
team_interfaces__msg__Emergency__init(team_interfaces__msg__Emergency * msg)
{
  if (!msg) {
    return false;
  }
  // emergency_stop
  // phase_change
  return true;
}

void
team_interfaces__msg__Emergency__fini(team_interfaces__msg__Emergency * msg)
{
  if (!msg) {
    return;
  }
  // emergency_stop
  // phase_change
}

bool
team_interfaces__msg__Emergency__are_equal(const team_interfaces__msg__Emergency * lhs, const team_interfaces__msg__Emergency * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // emergency_stop
  if (lhs->emergency_stop != rhs->emergency_stop) {
    return false;
  }
  // phase_change
  if (lhs->phase_change != rhs->phase_change) {
    return false;
  }
  return true;
}

bool
team_interfaces__msg__Emergency__copy(
  const team_interfaces__msg__Emergency * input,
  team_interfaces__msg__Emergency * output)
{
  if (!input || !output) {
    return false;
  }
  // emergency_stop
  output->emergency_stop = input->emergency_stop;
  // phase_change
  output->phase_change = input->phase_change;
  return true;
}

team_interfaces__msg__Emergency *
team_interfaces__msg__Emergency__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Emergency * msg = (team_interfaces__msg__Emergency *)allocator.allocate(sizeof(team_interfaces__msg__Emergency), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team_interfaces__msg__Emergency));
  bool success = team_interfaces__msg__Emergency__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team_interfaces__msg__Emergency__destroy(team_interfaces__msg__Emergency * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team_interfaces__msg__Emergency__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team_interfaces__msg__Emergency__Sequence__init(team_interfaces__msg__Emergency__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Emergency * data = NULL;

  if (size) {
    data = (team_interfaces__msg__Emergency *)allocator.zero_allocate(size, sizeof(team_interfaces__msg__Emergency), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team_interfaces__msg__Emergency__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team_interfaces__msg__Emergency__fini(&data[i - 1]);
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
team_interfaces__msg__Emergency__Sequence__fini(team_interfaces__msg__Emergency__Sequence * array)
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
      team_interfaces__msg__Emergency__fini(&array->data[i]);
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

team_interfaces__msg__Emergency__Sequence *
team_interfaces__msg__Emergency__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Emergency__Sequence * array = (team_interfaces__msg__Emergency__Sequence *)allocator.allocate(sizeof(team_interfaces__msg__Emergency__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team_interfaces__msg__Emergency__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team_interfaces__msg__Emergency__Sequence__destroy(team_interfaces__msg__Emergency__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team_interfaces__msg__Emergency__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team_interfaces__msg__Emergency__Sequence__are_equal(const team_interfaces__msg__Emergency__Sequence * lhs, const team_interfaces__msg__Emergency__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team_interfaces__msg__Emergency__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team_interfaces__msg__Emergency__Sequence__copy(
  const team_interfaces__msg__Emergency__Sequence * input,
  team_interfaces__msg__Emergency__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team_interfaces__msg__Emergency);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team_interfaces__msg__Emergency * data =
      (team_interfaces__msg__Emergency *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team_interfaces__msg__Emergency__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team_interfaces__msg__Emergency__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team_interfaces__msg__Emergency__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
