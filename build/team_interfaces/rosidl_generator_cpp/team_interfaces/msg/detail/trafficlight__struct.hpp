// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team_interfaces:msg/Trafficlight.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__STRUCT_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team_interfaces__msg__Trafficlight __attribute__((deprecated))
#else
# define DEPRECATED__team_interfaces__msg__Trafficlight __declspec(deprecated)
#endif

namespace team_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Trafficlight_
{
  using Type = Trafficlight_<ContainerAllocator>;

  explicit Trafficlight_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->traffic_light = false;
    }
  }

  explicit Trafficlight_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->traffic_light = false;
    }
  }

  // field types and members
  using _traffic_light_type =
    bool;
  _traffic_light_type traffic_light;

  // setters for named parameter idiom
  Type & set__traffic_light(
    const bool & _arg)
  {
    this->traffic_light = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team_interfaces::msg::Trafficlight_<ContainerAllocator> *;
  using ConstRawPtr =
    const team_interfaces::msg::Trafficlight_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Trafficlight_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Trafficlight_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team_interfaces__msg__Trafficlight
    std::shared_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team_interfaces__msg__Trafficlight
    std::shared_ptr<team_interfaces::msg::Trafficlight_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Trafficlight_ & other) const
  {
    if (this->traffic_light != other.traffic_light) {
      return false;
    }
    return true;
  }
  bool operator!=(const Trafficlight_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Trafficlight_

// alias to use template instance with default allocator
using Trafficlight =
  team_interfaces::msg::Trafficlight_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__TRAFFICLIGHT__STRUCT_HPP_
