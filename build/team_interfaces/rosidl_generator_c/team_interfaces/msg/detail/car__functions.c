// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from team_interfaces:msg/Car.idl
// generated code does not contain a copyright notice
#include "team_interfaces/msg/detail/car__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
team_interfaces__msg__Car__init(team_interfaces__msg__Car * msg)
{
  if (!msg) {
    return false;
  }
  // detected
  // distance
  return true;
}

void
team_interfaces__msg__Car__fini(team_interfaces__msg__Car * msg)
{
  if (!msg) {
    return;
  }
  // detected
  // distance
}

bool
team_interfaces__msg__Car__are_equal(const team_interfaces__msg__Car * lhs, const team_interfaces__msg__Car * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // detected
  if (lhs->detected != rhs->detected) {
    return false;
  }
  // distance
  if (lhs->distance != rhs->distance) {
    return false;
  }
  return true;
}

bool
team_interfaces__msg__Car__copy(
  const team_interfaces__msg__Car * input,
  team_interfaces__msg__Car * output)
{
  if (!input || !output) {
    return false;
  }
  // detected
  output->detected = input->detected;
  // distance
  output->distance = input->distance;
  return true;
}

team_interfaces__msg__Car *
team_interfaces__msg__Car__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Car * msg = (team_interfaces__msg__Car *)allocator.allocate(sizeof(team_interfaces__msg__Car), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(team_interfaces__msg__Car));
  bool success = team_interfaces__msg__Car__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
team_interfaces__msg__Car__destroy(team_interfaces__msg__Car * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    team_interfaces__msg__Car__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
team_interfaces__msg__Car__Sequence__init(team_interfaces__msg__Car__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Car * data = NULL;

  if (size) {
    data = (team_interfaces__msg__Car *)allocator.zero_allocate(size, sizeof(team_interfaces__msg__Car), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = team_interfaces__msg__Car__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        team_interfaces__msg__Car__fini(&data[i - 1]);
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
team_interfaces__msg__Car__Sequence__fini(team_interfaces__msg__Car__Sequence * array)
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
      team_interfaces__msg__Car__fini(&array->data[i]);
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

team_interfaces__msg__Car__Sequence *
team_interfaces__msg__Car__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  team_interfaces__msg__Car__Sequence * array = (team_interfaces__msg__Car__Sequence *)allocator.allocate(sizeof(team_interfaces__msg__Car__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = team_interfaces__msg__Car__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
team_interfaces__msg__Car__Sequence__destroy(team_interfaces__msg__Car__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    team_interfaces__msg__Car__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
team_interfaces__msg__Car__Sequence__are_equal(const team_interfaces__msg__Car__Sequence * lhs, const team_interfaces__msg__Car__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!team_interfaces__msg__Car__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
team_interfaces__msg__Car__Sequence__copy(
  const team_interfaces__msg__Car__Sequence * input,
  team_interfaces__msg__Car__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(team_interfaces__msg__Car);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    team_interfaces__msg__Car * data =
      (team_interfaces__msg__Car *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!team_interfaces__msg__Car__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          team_interfaces__msg__Car__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!team_interfaces__msg__Car__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
