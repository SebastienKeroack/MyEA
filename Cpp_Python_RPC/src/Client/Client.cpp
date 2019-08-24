#include "pch.hpp"

// Boost.
#include <boost/algorithm/string/replace.hpp>

// This.
#include <Client/Client.hpp>

// Common_Library.
#include <Strings/String.hpp>
#include <UI/Dialog_Box.hpp>

namespace MyEA::RPC
{
    Client::Client(void)
    {
    }
    
    void Client::Close(void)
    {
        if(py::extract<bool>(this->_client.attr("is_connected")()))
        {
            this->_client.attr("close")();
        }
    }
    
    bool Client::Initialized(void) const
    {
        return(Py_IsInitialized());
    }

    bool Client::Initialize(std::string const &script)
    {
        if(this->Initialized())
        {
            MyEA::String::Error("Initialization can only be call once.");
            
            return(false);
        }
        
        Py_Initialize();

        np::initialize();
        
        // |STR| Initialize main. |STR|
        this->_main   =       py::import("__main__");
        this->_global = this->_main.attr("__dict__");
        // |END| Initialize main. |END|
        
        // |STR| Prepare arguments. |STR|
        wchar_t **args = static_cast<wchar_t **>(PyMem_Malloc(3 * sizeof(wchar_t *)));
        
        args[0] = Py_DecodeLocale(boost::replace_all_copy(script, "\\", "\\\\").c_str(), NULL);
        args[1] = Py_DecodeLocale("--hosts", NULL);
        args[2] = Py_DecodeLocale("127.0.0.1=9000", NULL);
        
        PySys_SetArgv(3, args);
        // |END| Prepare arguments. |END|

        // Execute file.
        py::exec_file(script.c_str(),
                      this->_global,
                      this->_global);
        
        // Extract client object.
        this->_client = this->_main.attr("client");

        return(true);
    }
    
    bool Client::Open(void)
    {
        if(py::extract<bool>(this->_client.attr("open")()) == false)
        {
            MyEA::String::Error("An error has been triggered from the `open()` function.");
            
            return(false);
        }

        return(true);
    }
    
    np::ndarray Client::Predict(np::ndarray const &inputs)
    {
        auto result(Py_Call<np::ndarray>("Predict", this->_client,
                                         inputs));
        
        bool const &result_is_none(std::get<0>(result));
        
        if(result_is_none)
        {
            MyEA::String::Error("An error has been triggered from the `Predict()` function.");
            
            return(np::from_object(py::object()));
        }
        
        return(std::get<1>(result));
    }

    np::ndarray Client::Model_Metrics(void)
    {
        auto result(Py_Call<np::ndarray>("Model_Metrics", this->_client));
        
        bool const &result_is_none(std::get<0>(result));
        
        if(result_is_none)
        {
            MyEA::String::Error("An error has been triggered from the `Model_Metrics()` function.");
            
            return(np::from_object(py::object()));
        }
        
        return(std::get<1>(result));
    }

    Client::~Client(void)
    {
    }
}
