// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team_interfaces:msg/Car.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__CAR__STRUCT_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__CAR__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team_interfaces__msg__Car __attribute__((deprecated))
#else
# define DEPRECATED__team_interfaces__msg__Car __declspec(deprecated)
#endif

namespace team_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Car_
{
  using Type = Car_<ContainerAllocator>;

  explicit Car_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->detected = false;
      this->distance = 0.0f;
    }
  }

  explicit Car_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->detected = false;
      this->distance = 0.0f;
    }
  }

  // field types and members
  using _detected_type =
    bool;
  _detected_type detected;
  using _distance_type =
    float;
  _distance_type distance;

  // setters for named parameter idiom
  Type & set__detected(
    const bool & _arg)
  {
    this->detected = _arg;
    return *this;
  }
  Type & set__distance(
    const float & _arg)
  {
    this->distance = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team_interfaces::msg::Car_<ContainerAllocator> *;
  using ConstRawPtr =
    const team_interfaces::msg::Car_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team_interfaces::msg::Car_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team_interfaces::msg::Car_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Car_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Car_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Car_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Car_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team_interfaces::msg::Car_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team_interfaces::msg::Car_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team_interfaces__msg__Car
    std::shared_ptr<team_interfaces::msg::Car_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team_interfaces__msg__Car
    std::shared_ptr<team_interfaces::msg::Car_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Car_ & other) const
  {
    if (this->detected != other.detected) {
      return false;
    }
    if (this->distance != other.distance) {
      return false;
    }
    return true;
  }
  bool operator!=(const Car_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Car_

// alias to use template instance with default allocator
using Car =
  team_interfaces::msg::Car_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__CAR__STRUCT_HPP_
