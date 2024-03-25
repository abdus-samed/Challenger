# generated from rosidl_generator_py/resource/_idl.py.em
# with input from team_interfaces:msg/Signs.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Signs(type):
    """Metaclass of message 'Signs'."""

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
                'team_interfaces.msg.Signs')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__signs
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__signs
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__signs
            cls._TYPE_SUPPORT = module.type_support_msg__msg__signs
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__signs

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Signs(metaclass=Metaclass_Signs):
    """Message class 'Signs'."""

    __slots__ = [
        '_sign_detected',
        '_overtaking_allowed',
        '_overtaking_forbidden',
        '_cross_parking',
        '_parallel_parking',
        '_pit_in',
        '_pit_out',
        '_sign_height',
    ]

    _fields_and_field_types = {
        'sign_detected': 'boolean',
        'overtaking_allowed': 'boolean',
        'overtaking_forbidden': 'boolean',
        'cross_parking': 'boolean',
        'parallel_parking': 'boolean',
        'pit_in': 'boolean',
        'pit_out': 'boolean',
        'sign_height': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.sign_detected = kwargs.get('sign_detected', bool())
        self.overtaking_allowed = kwargs.get('overtaking_allowed', bool())
        self.overtaking_forbidden = kwargs.get('overtaking_forbidden', bool())
        self.cross_parking = kwargs.get('cross_parking', bool())
        self.parallel_parking = kwargs.get('parallel_parking', bool())
        self.pit_in = kwargs.get('pit_in', bool())
        self.pit_out = kwargs.get('pit_out', bool())
        self.sign_height = kwargs.get('sign_height', float())

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
        if self.sign_detected != other.sign_detected:
            return False
        if self.overtaking_allowed != other.overtaking_allowed:
            return False
        if self.overtaking_forbidden != other.overtaking_forbidden:
            return False
        if self.cross_parking != other.cross_parking:
            return False
        if self.parallel_parking != other.parallel_parking:
            return False
        if self.pit_in != other.pit_in:
            return False
        if self.pit_out != other.pit_out:
            return False
        if self.sign_height != other.sign_height:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def sign_detected(self):
        """Message field 'sign_detected'."""
        return self._sign_detected

    @sign_detected.setter
    def sign_detected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'sign_detected' field must be of type 'bool'"
        self._sign_detected = value

    @builtins.property
    def overtaking_allowed(self):
        """Message field 'overtaking_allowed'."""
        return self._overtaking_allowed

    @overtaking_allowed.setter
    def overtaking_allowed(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'overtaking_allowed' field must be of type 'bool'"
        self._overtaking_allowed = value

    @builtins.property
    def overtaking_forbidden(self):
        """Message field 'overtaking_forbidden'."""
        return self._overtaking_forbidden

    @overtaking_forbidden.setter
    def overtaking_forbidden(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'overtaking_forbidden' field must be of type 'bool'"
        self._overtaking_forbidden = value

    @builtins.property
    def cross_parking(self):
        """Message field 'cross_parking'."""
        return self._cross_parking

    @cross_parking.setter
    def cross_parking(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'cross_parking' field must be of type 'bool'"
        self._cross_parking = value

    @builtins.property
    def parallel_parking(self):
        """Message field 'parallel_parking'."""
        return self._parallel_parking

    @parallel_parking.setter
    def parallel_parking(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'parallel_parking' field must be of type 'bool'"
        self._parallel_parking = value

    @builtins.property
    def pit_in(self):
        """Message field 'pit_in'."""
        return self._pit_in

    @pit_in.setter
    def pit_in(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'pit_in' field must be of type 'bool'"
        self._pit_in = value

    @builtins.property
    def pit_out(self):
        """Message field 'pit_out'."""
        return self._pit_out

    @pit_out.setter
    def pit_out(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'pit_out' field must be of type 'bool'"
        self._pit_out = value

    @builtins.property
    def sign_height(self):
        """Message field 'sign_height'."""
        return self._sign_height

    @sign_height.setter
    def sign_height(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'sign_height' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'sign_height' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._sign_height = value
