# generated from rosidl_generator_py/resource/_idl.py.em
# with input from team_interfaces:msg/Lane.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'right_lane_x_points'
# Member 'right_lane_y_points'
# Member 'left_lane_x_points'
# Member 'left_lane_y_points'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Lane(type):
    """Metaclass of message 'Lane'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('team_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'team_interfaces.msg.Lane')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__lane
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__lane
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__lane
            cls._TYPE_SUPPORT = module.type_support_msg__msg__lane
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__lane

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Lane(metaclass=Metaclass_Lane):
    """Message class 'Lane'."""

    __slots__ = [
        '_right_lane_detected',
        '_left_lane_detected',
        '_dashed_line_detected',
        '_center_offset',
        '_heading_angle',
        '_right_lane_x_points',
        '_right_lane_y_points',
        '_left_lane_x_points',
        '_left_lane_y_points',
    ]

    _fields_and_field_types = {
        'right_lane_detected': 'boolean',
        'left_lane_detected': 'boolean',
        'dashed_line_detected': 'boolean',
        'center_offset': 'float',
        'heading_angle': 'float',
        'right_lane_x_points': 'sequence<float>',
        'right_lane_y_points': 'sequence<float>',
        'left_lane_x_points': 'sequence<float>',
        'left_lane_y_points': 'sequence<float>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.right_lane_detected = kwargs.get('right_lane_detected', bool())
        self.left_lane_detected = kwargs.get('left_lane_detected', bool())
        self.dashed_line_detected = kwargs.get('dashed_line_detected', bool())
        self.center_offset = kwargs.get('center_offset', float())
        self.heading_angle = kwargs.get('heading_angle', float())
        self.right_lane_x_points = array.array('f', kwargs.get('right_lane_x_points', []))
        self.right_lane_y_points = array.array('f', kwargs.get('right_lane_y_points', []))
        self.left_lane_x_points = array.array('f', kwargs.get('left_lane_x_points', []))
        self.left_lane_y_points = array.array('f', kwargs.get('left_lane_y_points', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.right_lane_detected != other.right_lane_detected:
            return False
        if self.left_lane_detected != other.left_lane_detected:
            return False
        if self.dashed_line_detected != other.dashed_line_detected:
            return False
        if self.center_offset != other.center_offset:
            return False
        if self.heading_angle != other.heading_angle:
            return False
        if self.right_lane_x_points != other.right_lane_x_points:
            return False
        if self.right_lane_y_points != other.right_lane_y_points:
            return False
        if self.left_lane_x_points != other.left_lane_x_points:
            return False
        if self.left_lane_y_points != other.left_lane_y_points:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def right_lane_detected(self):
        """Message field 'right_lane_detected'."""
        return self._right_lane_detected

    @right_lane_detected.setter
    def right_lane_detected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'right_lane_detected' field must be of type 'bool'"
        self._right_lane_detected = value

    @builtins.property
    def left_lane_detected(self):
        """Message field 'left_lane_detected'."""
        return self._left_lane_detected

    @left_lane_detected.setter
    def left_lane_detected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'left_lane_detected' field must be of type 'bool'"
        self._left_lane_detected = value

    @builtins.property
    def dashed_line_detected(self):
        """Message field 'dashed_line_detected'."""
        return self._dashed_line_detected

    @dashed_line_detected.setter
    def dashed_line_detected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'dashed_line_detected' field must be of type 'bool'"
        self._dashed_line_detected = value

    @builtins.property
    def center_offset(self):
        """Message field 'center_offset'."""
        return self._center_offset

    @center_offset.setter
    def center_offset(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'center_offset' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'center_offset' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._center_offset = value

    @builtins.property
    def heading_angle(self):
        """Message field 'heading_angle'."""
        return self._heading_angle

    @heading_angle.setter
    def heading_angle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'heading_angle' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'heading_angle' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._heading_angle = value

    @builtins.property
    def right_lane_x_points(self):
        """Message field 'right_lane_x_points'."""
        return self._right_lane_x_points

    @right_lane_x_points.setter
    def right_lane_x_points(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'right_lane_x_points' array.array() must have the type code of 'f'"
            self._right_lane_x_points = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'right_lane_x_points' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._right_lane_x_points = array.array('f', value)

    @builtins.property
    def right_lane_y_points(self):
        """Message field 'right_lane_y_points'."""
        return self._right_lane_y_points

    @right_lane_y_points.setter
    def right_lane_y_points(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'right_lane_y_points' array.array() must have the type code of 'f'"
            self._right_lane_y_points = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'right_lane_y_points' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._right_lane_y_points = array.array('f', value)

    @builtins.property
    def left_lane_x_points(self):
        """Message field 'left_lane_x_points'."""
        return self._left_lane_x_points

    @left_lane_x_points.setter
    def left_lane_x_points(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'left_lane_x_points' array.array() must have the type code of 'f'"
            self._left_lane_x_points = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'left_lane_x_points' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._left_lane_x_points = array.array('f', value)

    @builtins.property
    def left_lane_y_points(self):
        """Message field 'left_lane_y_points'."""
        return self._left_lane_y_points

    @left_lane_y_points.setter
    def left_lane_y_points(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'left_lane_y_points' array.array() must have the type code of 'f'"
            self._left_lane_y_points = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'left_lane_y_points' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._left_lane_y_points = array.array('f', value)
