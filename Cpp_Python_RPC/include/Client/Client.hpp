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

            bool Initialize(std::string const &script);

            bool Open(void);
            
            np::ndarray Merge_X(np::ndarray const &inputs);

            np::ndarray Merge_Y(np::ndarray const &inputs);

            np::ndarray Predict(np::ndarray const &inputs);
            
            np::ndarray Model_Metrics(void);

        private:
            py::object _main  ;
            py::object _global;
            py::object _client;
    };
}