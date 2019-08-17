#include "pch.hpp"

// This.
#include <API.hpp>
#include <Client/Client.hpp>

// Common_Library.
#include <Strings/String.hpp>

class MyEA::RPC::Client g_Client;

template<class Fn, class ... Args>
auto Py_Try(Fn&& fn_received, Args&& ... args)
{
    bool tmp_success(true);

    if constexpr (std::is_same_v<std::invoke_result_t<Fn, Args...>, void>)
    {
        try
        {
            //  invoke: Call a function or a member function from args[0].
            // forward: forward rvalue.
            std::invoke(std::forward<Fn>(fn_received), std::forward<Args>(args)...);
        }
        catch(py::error_already_set &)
        {
            tmp_success = false;
        
            MyEA::String::Error(PyErr_Parse().c_str());
        }
    
        return(tmp_success);
    }
    else
    {
        std::invoke_result_t<Fn, Args...> tmp_fn_result;

        try
        {
            //  invoke: Call a function or a member function from args[0].
            // forward: forward rvalue.
            tmp_fn_result = std::invoke(std::forward<Fn>(fn_received), std::forward<Args>(args)...);
        }
        catch(py::error_already_set &)
        {
            tmp_success = false;
            
            MyEA::String::Error(PyErr_Parse().c_str());
        }
    
        return(std::tuple(tmp_success, tmp_fn_result));
    }
}

DLL_API bool API__Cpp_Python_RPC__Initialize(void)
{
    if(g_Client.Initialized())
    {
        MyEA::String::Error("Initialization can only be call once per load." NEW_LINE
                            "Unload the `.dll` and retry.");
        
        return(false);
    }

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Initialize, std::ref(g_Client),
                                  "C:\\Users\\sebas\\Documents\\MEGAsync\\MyEA\\Python\\run_client.py"));
    
    // Try success.
    if(std::get<0>(tmp_results) == false)
    {
        MyEA::String::Error("An error has been triggered from the `Initialize()` function.");
        
        return(false);
    }
    // Called fn success.
    else if(std::get<1>(tmp_results))
    {
        MyEA::String::Error("Initialization can only be call once per load." NEW_LINE
                            "Unload the `.dll` and retry.");
        
        return(false);
    }

    return(true);
}

DLL_API bool API__Cpp_Python_RPC__Call(void)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::String::Error("Client is not initialized.");
        
        return(false);
    }
    else if(Py_Try(&MyEA::RPC::Client::Call, std::ref(g_Client)) == false)
    {
        MyEA::String::Error("An error has been triggered from the `Call()` function.");
        
        return(false);
    }
    
    return(true);
}
