#include "pch.hpp"

// Boost.
#include <boost/python.hpp>

namespace py = boost::python;

// This.
#include <Client/Client.hpp>

// Common_Library.
#include <UI/Dialog_Box.hpp>

namespace MyEA::RPC
{
    Client::Client(void)
    {
    }
    
    void Client::foo(void)
    {
        static int aaa = 0;

        DEBUG_BOX("Hello world! #" + std::to_string(aaa++))

        py::object main_module = py::import("__main__");
        py::object main_namespace = main_module.attr("__dict__");
        
        std::string const tmp_command("result = 5 ** " + std::to_string(aaa));
        py::object ignored = exec(tmp_command.c_str(), main_namespace);
        int five_squared = py::extract<int>(main_namespace["result"]);
        
        DEBUG_BOX("five_squared=" + std::to_string(five_squared))

        /*
        DEBUG_BOX("Result=" + std::to_string(five_squared));

        py::object ignored = py::exec("hello = file('hello.txt', 'w')\n"
                                      "hello.write('Hello world!')\n"
                                      "hello.close()",
                                      main_namespace);

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
