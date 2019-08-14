#include <pch.hpp>

// Boost.
#include <boost/python.hpp>

namespace py = boost::python;

// This.
#include <Client/Client.hpp>

namespace MyEA::RPC
{
    Client::Client(void)
    {
    }
    
    void Client::foo(void)
    {
        py::object main_module = py::import("__main__");
        py::object main_namespace = main_module.attr("__dict__");

        py::object ignored = py::exec("hello = file('hello.txt', 'w')\n"
                                      "hello.write('Hello world!')\n"
                                      "hello.close()",
                                      main_namespace);

        /*
        py::object main_module = py::import("__main__");
        py::object main_namespace = main_module.attr("__dict__");
        py::object ignored = exec("result = 5 ** 2", main_namespace);
        int five_squared = py::extract<int>(main_namespace["result"]);
        */
    }

    Client::~Client(void)
    {
    }
}
