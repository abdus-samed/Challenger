// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from team_interfaces:msg/Lane.idl
// generated code does not contain a copyright notice

#ifndef TEAM_INTERFACES__MSG__DETAIL__LANE__STRUCT_HPP_
#define TEAM_INTERFACES__MSG__DETAIL__LANE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__team_interfaces__msg__Lane __attribute__((deprecated))
#else
# define DEPRECATED__team_interfaces__msg__Lane __declspec(deprecated)
#endif

namespace team_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Lane_
{
  using Type = Lane_<ContainerAllocator>;

  explicit Lane_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->right_lane_detected = false;
      this->left_lane_detected = false;
      this->dashed_line_detected = false;
      this->center_offset = 0.0f;
      this->heading_angle = 0.0f;
    }
  }

  explicit Lane_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->right_lane_detected = false;
      this->left_lane_detected = false;
      this->dashed_line_detected = false;
      this->center_offset = 0.0f;
      this->heading_angle = 0.0f;
    }
  }

  // field types and members
  using _right_lane_detected_type =
    bool;
  _right_lane_detected_type right_lane_detected;
  using _left_lane_detected_type =
    bool;
  _left_lane_detected_type left_lane_detected;
  using _dashed_line_detected_type =
    bool;
  _dashed_line_detected_type dashed_line_detected;
  using _center_offset_type =
    float;
  _center_offset_type center_offset;
  using _heading_angle_type =
    float;
  _heading_angle_type heading_angle;
  using _right_lane_x_points_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _right_lane_x_points_type right_lane_x_points;
  using _right_lane_y_points_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _right_lane_y_points_type right_lane_y_points;
  using _left_lane_x_points_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _left_lane_x_points_type left_lane_x_points;
  using _left_lane_y_points_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _left_lane_y_points_type left_lane_y_points;

  // setters for named parameter idiom
  Type & set__right_lane_detected(
    const bool & _arg)
  {
    this->right_lane_detected = _arg;
    return *this;
  }
  Type & set__left_lane_detected(
    const bool & _arg)
  {
    this->left_lane_detected = _arg;
    return *this;
  }
  Type & set__dashed_line_detected(
    const bool & _arg)
  {
    this->dashed_line_detected = _arg;
    return *this;
  }
  Type & set__center_offset(
    const float & _arg)
  {
    this->center_offset = _arg;
    return *this;
  }
  Type & set__heading_angle(
    const float & _arg)
  {
    this->heading_angle = _arg;
    return *this;
  }
  Type & set__right_lane_x_points(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->right_lane_x_points = _arg;
    return *this;
  }
  Type & set__right_lane_y_points(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->right_lane_y_points = _arg;
    return *this;
  }
  Type & set__left_lane_x_points(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->left_lane_x_points = _arg;
    return *this;
  }
  Type & set__left_lane_y_points(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->left_lane_y_points = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    team_interfaces::msg::Lane_<ContainerAllocator> *;
  using ConstRawPtr =
    const team_interfaces::msg::Lane_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<team_interfaces::msg::Lane_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<team_interfaces::msg::Lane_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Lane_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Lane_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      team_interfaces::msg::Lane_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<team_interfaces::msg::Lane_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<team_interfaces::msg::Lane_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<team_interfaces::msg::Lane_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__team_interfaces__msg__Lane
    std::shared_ptr<team_interfaces::msg::Lane_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__team_interfaces__msg__Lane
    std::shared_ptr<team_interfaces::msg::Lane_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Lane_ & other) const
  {
    if (this->right_lane_detected != other.right_lane_detected) {
      return false;
    }
    if (this->left_lane_detected != other.left_lane_detected) {
      return false;
    }
    if (this->dashed_line_detected != other.dashed_line_detected) {
      return false;
    }
    if (this->center_offset != other.center_offset) {
      return false;
    }
    if (this->heading_angle != other.heading_angle) {
      return false;
    }
    if (this->right_lane_x_points != other.right_lane_x_points) {
      return false;
    }
    if (this->right_lane_y_points != other.right_lane_y_points) {
      return false;
    }
    if (this->left_lane_x_points != other.left_lane_x_points) {
      return false;
    }
    if (this->left_lane_y_points != other.left_lane_y_points) {
      return false;
    }
    return true;
  }
  bool operator!=(const Lane_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Lane_

// alias to use template instance with default allocator
using Lane =
  team_interfaces::msg::Lane_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace team_interfaces

#endif  // TEAM_INTERFACES__MSG__DETAIL__LANE__STRUCT_HPP_
