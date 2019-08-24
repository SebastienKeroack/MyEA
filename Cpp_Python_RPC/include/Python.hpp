#pragma once

// Boost.
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

std::string PyErr_Parse();

#include <UI/Dialog_Box.hpp> // WARNING

template<class Fn, class ... Args>
auto Py_Try(Fn&& fn_received, Args&& ... args)
{
    typedef std::invoke_result_t<Fn, Args...> result_t;
    
    bool tmp_success(true);

    if constexpr (std::is_same_v<result_t, void>)
    {
        try
        {
            //  invoke: Call a function or a member function from args[0].
            // forward: forward rvalue.
            std::invoke(std::forward<Fn>(fn_received), std::forward<Args>(args)...);
        }
        catch(py::error_already_set const &)
        {
            DEBUG_BOX(PyErr_Parse().c_str());

            tmp_success = false;
        }
    
        return(tmp_success);
    }
    else if constexpr (std::is_same_v<result_t, np::ndarray>)
    {
        try
        {
            return(std::tuple(true, std::invoke(std::forward<Fn>(fn_received), std::forward<Args>(args)...)));
        }
        catch(py::error_already_set const &)
        {
            DEBUG_BOX(PyErr_Parse().c_str());

            return(std::tuple(false, np::from_object(py::object())));
        }
    }
    else
    {
        try
        {
            return(std::tuple(true, std::invoke(std::forward<Fn>(fn_received), std::forward<Args>(args)...)));
        }
        catch(py::error_already_set const &)
        {
            DEBUG_BOX(PyErr_Parse().c_str());

            return(std::tuple(false, result_t()));
        }
    }
}

template<typename T, class ... Args>
std::tuple<bool, T> Py_Call(char const *fn_name_received,
                            py::object &ref_caller_received,
                            Args&& ... args)
{
    py::object tmp_result_as_object(ref_caller_received.attr(fn_name_received)(args...));
        
    bool const tmp_is_none(tmp_result_as_object.is_none());

    if constexpr (std::is_same_v<T, np::ndarray>)
    {
        return(std::tuple(tmp_is_none, np::from_object(tmp_result_as_object)));
    }
    else
    {
        return(std::tuple(tmp_is_none, py::extract<T>(tmp_result_as_object)));
    }
}
