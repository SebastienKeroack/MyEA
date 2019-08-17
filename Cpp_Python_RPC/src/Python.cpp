#include "pch.hpp"

// This.
#include <Python.hpp>

std::string PyErr_Parse()
{
    PyObject *tmp_ptr_extype,
             *tmp_ptr_value,
             *tmp_ptr_traceback;
    
    PyErr_Fetch(&tmp_ptr_extype,
                &tmp_ptr_value,
                &tmp_ptr_traceback);
    
    if(tmp_ptr_extype == NULL) { return(""); }

    py::object const tmp_extype   (py::handle<>(py::borrowed(tmp_ptr_extype   ))),
                     tmp_value    (py::handle<>(py::borrowed(tmp_ptr_value    ))),
                     tmp_traceback(py::handle<>(py::borrowed(tmp_ptr_traceback))),
                     tmp_import_traceback(py::import("traceback")),
                     tmp_lines(tmp_import_traceback.attr("format_exception")(tmp_extype,
                                                                             tmp_value,
                                                                             tmp_traceback));
    
    std::string tmp_output("");

    for(int i(0); i != py::len(tmp_lines); ++i)
    {
        tmp_output += py::extract<std::string>(tmp_lines[i])();
    }

    // PyErr_Fetch clears the error state, uncomment
    // the following line to restore the error state:
    // PyErr_Restore(extype, value, traceback);

    return(tmp_output);
}
