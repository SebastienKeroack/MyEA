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

    py::object const extype   (py::handle<>(py::borrowed(ptr_extype   ))),
                     value    (py::handle<>(py::borrowed(ptr_value    ))),
                     traceback(py::handle<>(py::borrowed(ptr_traceback))),
                     import_traceback(py::import("traceback")),
                     lines(import_traceback.attr("format_exception")(extype,
                                                                     value,
                                                                     traceback));
    
    std::string outputs("");

    for(int i(0); i != py::len(lines); ++i)
    {
        outputs += py::extract<std::string>(lines[i])();
    }

    // PyErr_Fetch clears the error state, uncomment
    // the following line to restore the error state:
    // PyErr_Restore(extype, value, traceback);

    return(outputs);
}
