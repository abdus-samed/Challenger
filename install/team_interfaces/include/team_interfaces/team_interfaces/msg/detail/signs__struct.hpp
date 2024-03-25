// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__SIGNS__STRUCT_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__SIGNS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team_interfaces__msg__Signs __attribute__((deprecated))
#else
# define DEPRECATED__team_interfaces__msg__Signs __declspec(deprecated)
#endif

namespace team_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Signs_
{
  using Type = Signs_<ContainerAllocator>;

  explicit Signs_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sign_detected = false;
      this->overtaking_allowed = false;
      this->overtaking_forbidden = false;
      this->cross_parking = false;
      this->parallel_parking = false;
      this->pit_in = false;
      this->pit_out = false;
      this->sign_height = 0.0f;
    }
  }

  explicit Signs_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sign_detected = false;
      this->overtaking_allowed = false;
      this->overtaking_forbidden = false;
      this->cross_parking = false;
      this->parallel_parking = false;
      this->pit_in = false;
      this->pit_out = false;
      this->sign_height = 0.0f;
    }
  }

  // field types and members
  using _sign_detected_type =
    bool;
  _sign_detected_type sign_detected;
  using _overtaking_allowed_type =
    bool;
  _overtaking_allowed_type overtaking_allowed;
  using _overtaking_forbidden_type =
    bool;
  _overtaking_forbidden_type overtaking_forbidden;
  using _cross_parking_type =
    bool;
  _cross_parking_type cross_parking;
  using _parallel_parking_type =
    bool;
  _parallel_parking_type parallel_parking;
  using _pit_in_type =
    bool;
  _pit_in_type pit_in;
  using _pit_out_type =
    bool;
  _pit_out_type pit_out;
  using _sign_height_type =
    float;
  _sign_height_type sign_height;

  // setters for named parameter idiom
  Type & set__sign_detected(
    const bool & _arg)
  {
    this->sign_detected = _arg;
    return *this;
  }
  Type & set__overtaking_allowed(
    const bool & _arg)
  {
    this->overtaking_allowed = _arg;
    return *this;
  }
  Type & set__overtaking_forbidden(
    const bool & _arg)
  {
    this->overtaking_forbidden = _arg;
    return *this;
  }
  Type & set__cross_parking(
    const bool & _arg)
  {
    this->cross_parking = _arg;
    return *this;
  }
  Type & set__parallel_parking(
    const bool & _arg)
  {
    this->parallel_parking = _arg;
    return *this;
  }
  Type & set__pit_in(
    const bool & _arg)
  {
    this->pit_in = _arg;
    return *this;
  }
  Type & set__pit_out(
    const bool & _arg)
  {
    this->pit_out = _arg;
    return *this;
  }
  Type & set__sign_height(
    const float & _arg)
  {
    this->sign_height = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team_interfaces::msg::Signs_<ContainerAllocator> *;
  using ConstRawPtr =
    const team_interfaces::msg::Signs_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team_interfaces::msg::Signs_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team_interfaces::msg::Signs_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Signs_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Signs_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Signs_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Signs_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team_interfaces::msg::Signs_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team_interfaces::msg::Signs_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team_interfaces__msg__Signs
    std::shared_ptr<team_interfaces::msg::Signs_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team_interfaces__msg__Signs
    std::shared_ptr<team_interfaces::msg::Signs_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Signs_ & other) const
  {
    if (this->sign_detected != other.sign_detected) {
      return false;
    }
    if (this->overtaking_allowed != other.overtaking_allowed) {
      return false;
    }
    if (this->overtaking_forbidden != other.overtaking_forbidden) {
      return false;
    }
    if (this->cross_parking != other.cross_parking) {
      return false;
    }
    if (this->parallel_parking != other.parallel_parking) {
      return false;
    }
    if (this->pit_in != other.pit_in) {
      return false;
    }
    if (this->pit_out != other.pit_out) {
      return false;
    }
    if (this->sign_height != other.sign_height) {
      return false;
    }
    return true;
  }
  bool operator!=(const Signs_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Signs_

// alias to use template instance with default allocator
using Signs =
  team_interfaces::msg::Signs_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__SIGNS__STRUCT_HPP_
