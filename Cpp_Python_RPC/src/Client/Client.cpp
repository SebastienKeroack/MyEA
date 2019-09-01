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
        if(py::extract<bool>(this->_client_opt.attr("is_connected")()))
        {
            this->_client_opt.attr("close")();
        }

        if(py::extract<bool>(this->_client_inf.attr("is_connected")()))
        {
            this->_client_inf.attr("close")();
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
        // |END| Prepare arguments. |END|
        
        auto Construct_Client([this, &args, &script](int const port)
        {
            // |STR| Prepare arguments. |STR|
            args[2] = Py_DecodeLocale(std::string("192.168.1.100=" + std::to_string(port)).c_str(), NULL);
            
            PySys_SetArgv(3, args);
            // |END| Prepare arguments. |END|
            
            // Execute file.
            py::exec_file(script.c_str(),
                          this->_global,
                          this->_global);
            
            // Extract client object.
            return(this->_main.attr("client"));
        });

        this->_client_opt = Construct_Client(9000);
        this->_client_inf = Construct_Client(9001);

        return(true);
    }
    
    bool Client::Open(void)
    {
        if(py::extract<bool>(this->_client_opt.attr("open")()) == false)
        {
            MyEA::String::Error("An error has been triggered from the `open()` function.");
            
            return(false);
        }
        
        if(py::extract<bool>(this->_client_inf.attr("open")()) == false)
        {
            MyEA::String::Error("An error has been triggered from the `open()` function.");
            
            return(false);
        }

        return(true);
    }
    
    np::ndarray Client::Concatenate_X(np::ndarray const &inputs)
    {
        auto result(Py_Call<np::ndarray>("Concatenate_X", this->_client_opt,
                                         inputs));
        
        bool const &result_is_none(std::get<0>(result));
        
        if(result_is_none)
        {
            MyEA::String::Error("An error has been triggered from the `Concatenate_X()` function.");
            
            return(np::from_object(py::object()));
        }
        
        return(std::get<1>(result));
    }
    
    np::ndarray Client::Concatenate_Y(np::ndarray const &inputs)
    {
        auto result(Py_Call<np::ndarray>("Concatenate_Y", this->_client_opt,
                                         inputs));
        
        bool const &result_is_none(std::get<0>(result));
        
        if(result_is_none)
        {
            MyEA::String::Error("An error has been triggered from the `Concatenate_Y()` function.");
            
            return(np::from_object(py::object()));
        }
        
        return(std::get<1>(result));
    }

    np::ndarray Client::Predict(py::list const &inputs)
    {
        auto result(Py_Call<np::ndarray>("Predict", this->_client_inf,
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
        auto result(Py_Call<np::ndarray>("Model_Metrics", this->_client_opt));
        
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
