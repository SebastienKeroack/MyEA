#pragma once

// Common_Library.
#include <Configuration/dtypes.hpp>

// Standard.
#include <string>

// This.
#include <Python.hpp>

namespace MyEA::RPC
{
    class Client
    {
        public:
            Client(void);
            
            ~Client(void);
            
            void Close(void);

            bool Initialized(void) const;

            bool Initialize(std::string const &ref_script_path_received);

            bool Open(void);
            
            T_ Predict(void) const;

        private:
            py::object _main  ;
            py::object _global;
            py::object _client;
    };
}