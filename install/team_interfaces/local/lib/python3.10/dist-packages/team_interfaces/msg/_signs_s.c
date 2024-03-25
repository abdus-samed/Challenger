// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from team_interfaces:msg/Signs.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "team_interfaces/msg/detail/signs__struct.h"
#include "team_interfaces/msg/detail/signs__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool team_interfaces__msg__signs__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[33];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("team_interfaces.msg._signs.Signs", full_classname_dest, 32) == 0);
  }
  team_interfaces__msg__Signs * ros_message = _ros_message;
  {  // sign_detected
    PyObject * field = PyObject_GetAttrString(_pymsg, "sign_detected");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->sign_detected = (Py_True == field);
    Py_DECREF(field);
  }
  {  // overtaking_allowed
    PyObject * field = PyObject_GetAttrString(_pymsg, "overtaking_allowed");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->overtaking_allowed = (Py_True == field);
    Py_DECREF(field);
  }
  {  // overtaking_forbidden
    PyObject * field = PyObject_GetAttrString(_pymsg, "overtaking_forbidden");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->overtaking_forbidden = (Py_True == field);
    Py_DECREF(field);
  }
  {  // cross_parking
    PyObject * field = PyObject_GetAttrString(_pymsg, "cross_parking");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->cross_parking = (Py_True == field);
    Py_DECREF(field);
  }
  {  // parallel_parking
    PyObject * field = PyObject_GetAttrString(_pymsg, "parallel_parking");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->parallel_parking = (Py_True == field);
    Py_DECREF(field);
  }
  {  // pit_in
    PyObject * field = PyObject_GetAttrString(_pymsg, "pit_in");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->pit_in = (Py_True == field);
    Py_DECREF(field);
  }
  {  // pit_out
    PyObject * field = PyObject_GetAttrString(_pymsg, "pit_out");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->pit_out = (Py_True == field);
    Py_DECREF(field);
  }
  {  // sign_height
    PyObject * field = PyObject_GetAttrString(_pymsg, "sign_height");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->sign_height = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * team_interfaces__msg__signs__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Signs */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("team_interfaces.msg._signs");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Signs");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  team_interfaces__msg__Signs * ros_message = (team_interfaces__msg__Signs *)raw_ros_message;
  {  // sign_detected
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->sign_detected ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "sign_detected", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // overtaking_allowed
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->overtaking_allowed ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "overtaking_allowed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // overtaking_forbidden
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->overtaking_forbidden ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "overtaking_forbidden", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cross_parking
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->cross_parking ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cross_parking", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // parallel_parking
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->parallel_parking ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "parallel_parking", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pit_in
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->pit_in ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pit_in", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pit_out
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->pit_out ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pit_out", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // sign_height
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->sign_height);
    {
      int rc = PyObject_SetAttrString(_pymessage, "sign_height", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
