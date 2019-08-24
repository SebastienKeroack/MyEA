#include "pch.hpp"

// This.
#include <Python.hpp>

std::string PyErr_Parse()
{
    PyObject *ptr_extype,
             *ptr_value,
             *ptr_traceback;
    
    PyErr_Fetch(&ptr_extype,
                &ptr_value,
                &ptr_traceback);
    
    if(ptr_extype == NULL) { return(""); }
    
    py::object const tmp_traceback(py::handle<>(py::borrowed(ptr_traceback)));
    
    long const at_line = py::extract<long>(tmp_traceback.attr("tb_lineno"));

    std::string const at_filename = py::extract<std::string>(tmp_traceback.attr("tb_frame").attr("f_code").attr("co_filename")),
                      cause = py::extract<std::string>(ptr_value);
    
    return(at_filename + ":" + std::to_string(at_line) + ", " + cause);
}
