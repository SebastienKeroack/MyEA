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

            bool Initialize(std::string const &host, std::string const &script);

            bool Open(void);
            
            bool Concatenate_X(np::ndarray const &inputs);

            bool Concatenate_Y(np::ndarray const &inputs);

            np::ndarray Normalize_X(np::ndarray const &inputs, bool const fixed = false);

            np::ndarray Predict(py::list const &inputs);
            
            np::ndarray Model_Metrics(void);

        private:
            py::object _main  ;
            py::object _global;
            py::object _client_opt;
            py::object _client_inf;
    };
}