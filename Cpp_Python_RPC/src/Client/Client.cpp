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
    
    bool Client::Initialized(void) const
    {
        return(Py_IsInitialized());
    }

    bool Client::Initialize(std::string const &ref_script_path_received)
    {
        if(this->Initialized())
        {
            MyEA::String::Error("Initialization can only be call once.");
            
            return(false);
        }
        
        Py_Initialize();
        
        // |STR| Initialize main. |STR|
        this->_main   =       py::import("__main__");
        this->_global = this->_main.attr("__dict__");
        // |END| Initialize main. |END|
        
        // |STR| Prepare arguments. |STR|
        wchar_t **tmp_args = static_cast<wchar_t **>(PyMem_Malloc(3 * sizeof(wchar_t *)));
            
        tmp_args[0] = Py_DecodeLocale(boost::replace_all_copy(ref_script_path_received, "\\", "\\\\").c_str(), NULL);
        tmp_args[1] = Py_DecodeLocale("--hosts", NULL);
        tmp_args[2] = Py_DecodeLocale("127.0.0.1=9000", NULL);
            
        PySys_SetArgv(3, tmp_args);
        // |END| Prepare arguments. |END|

        // Execute file.
        py::exec_file(ref_script_path_received.c_str(),
                      this->_global,
                      this->_global);
        
        // Extract client object.
        this->_client = this->_main.attr("client");

        return(true);
    }

    void Client::Call(void)
    {
        DEBUG_BOX("Hello world!");

        bool const success(py::extract<bool>(this->_client.attr("open")()));

        DEBUG_BOX("Success=" + std::to_string(success))

        DEBUG_BOX("Bye world!");
    }

    Client::~Client(void)
    {
    }
}
