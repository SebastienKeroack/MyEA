#pragma once

#include <Python.hpp>
#include <string>

namespace MyEA::RPC
{
    class Client
    {
        public:
            Client(void);
            
            ~Client(void);

            bool Initialized(void) const;

            bool Initialize(std::string const &ref_script_path_received);

            void Call(void);

        private:
            py::object _main  ;
            py::object _global;
            py::object _client;
    };
}