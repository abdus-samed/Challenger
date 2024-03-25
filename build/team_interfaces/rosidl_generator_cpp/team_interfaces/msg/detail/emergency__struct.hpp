// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team_interfaces:msg/Emergency.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__STRUCT_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team_interfaces__msg__Emergency __attribute__((deprecated))
#else
# define DEPRECATED__team_interfaces__msg__Emergency __declspec(deprecated)
#endif

namespace team_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Emergency_
{
  using Type = Emergency_<ContainerAllocator>;

  explicit Emergency_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->emergency_stop = false;
      this->phase_change = false;
    }
  }

  explicit Emergency_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->emergency_stop = false;
      this->phase_change = false;
    }
  }

  // field types and members
  using _emergency_stop_type =
    bool;
  _emergency_stop_type emergency_stop;
  using _phase_change_type =
    bool;
  _phase_change_type phase_change;

  // setters for named parameter idiom
  Type & set__emergency_stop(
    const bool & _arg)
  {
    this->emergency_stop = _arg;
    return *this;
  }
  Type & set__phase_change(
    const bool & _arg)
  {
    this->phase_change = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team_interfaces::msg::Emergency_<ContainerAllocator> *;
  using ConstRawPtr =
    const team_interfaces::msg::Emergency_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team_interfaces::msg::Emergency_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team_interfaces::msg::Emergency_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Emergency_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Emergency_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Emergency_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Emergency_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team_interfaces::msg::Emergency_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team_interfaces::msg::Emergency_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team_interfaces__msg__Emergency
    std::shared_ptr<team_interfaces::msg::Emergency_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team_interfaces__msg__Emergency
    std::shared_ptr<team_interfaces::msg::Emergency_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Emergency_ & other) const
  {
    if (this->emergency_stop != other.emergency_stop) {
      return false;
    }
    if (this->phase_change != other.phase_change) {
      return false;
    }
    return true;
  }
  bool operator!=(const Emergency_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Emergency_

// alias to use template instance with default allocator
using Emergency =
  team_interfaces::msg::Emergency_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__EMERGENCY__STRUCT_HPP_
