// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__LANE__FUNCTIONS_H_
#define TEAM_INTERFACES__MSG__DETAIL__LANE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "team_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "team_interfaces/msg/detail/lane__struct.h"

/// Initialize msg/Lane message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * team_interfaces__msg__Lane
 * )) before or use
 * team_interfaces__msg__Lane__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
bool
team_interfaces__msg__Lane__init(team_interfaces__msg__Lane * msg);

/// Finalize msg/Lane message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
void
team_interfaces__msg__Lane__fini(team_interfaces__msg__Lane * msg);

/// Create msg/Lane message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * team_interfaces__msg__Lane__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
team_interfaces__msg__Lane *
team_interfaces__msg__Lane__create();

/// Destroy msg/Lane message.
/**
 * It calls
 * team_interfaces__msg__Lane__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
void
team_interfaces__msg__Lane__destroy(team_interfaces__msg__Lane * msg);

/// Check for msg/Lane message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
bool
team_interfaces__msg__Lane__are_equal(const team_interfaces__msg__Lane * lhs, const team_interfaces__msg__Lane * rhs);

/// Copy a msg/Lane message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
bool
team_interfaces__msg__Lane__copy(
  const team_interfaces__msg__Lane * input,
  team_interfaces__msg__Lane * output);

/// Initialize array of msg/Lane messages.
/**
 * It allocates the memory for the number of elements and calls
 * team_interfaces__msg__Lane__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
bool
team_interfaces__msg__Lane__Sequence__init(team_interfaces__msg__Lane__Sequence * array, size_t size);

/// Finalize array of msg/Lane messages.
/**
 * It calls
 * team_interfaces__msg__Lane__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
void
team_interfaces__msg__Lane__Sequence__fini(team_interfaces__msg__Lane__Sequence * array);

/// Create array of msg/Lane messages.
/**
 * It allocates the memory for the array and calls
 * team_interfaces__msg__Lane__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
team_interfaces__msg__Lane__Sequence *
team_interfaces__msg__Lane__Sequence__create(size_t size);

/// Destroy array of msg/Lane messages.
/**
 * It calls
 * team_interfaces__msg__Lane__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
void
team_interfaces__msg__Lane__Sequence__destroy(team_interfaces__msg__Lane__Sequence * array);

/// Check for msg/Lane message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
bool
team_interfaces__msg__Lane__Sequence__are_equal(const team_interfaces__msg__Lane__Sequence * lhs, const team_interfaces__msg__Lane__Sequence * rhs);

/// Copy an array of msg/Lane messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_team_interfaces
bool
team_interfaces__msg__Lane__Sequence__copy(
  const team_interfaces__msg__Lane__Sequence * input,
  team_interfaces__msg__Lane__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // TEAM_INTERFACES__MSG__DETAIL__LANE__FUNCTIONS_H_
