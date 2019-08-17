#include "pch.hpp"

// This.
#include <API.hpp>
#include <Client/Client.hpp>

// Common_Library.
#include <Files/File.hpp>
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
        
            MyEA::File::fError(PyErr_Parse().c_str());
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
            
            MyEA::File::fError(PyErr_Parse().c_str());
        }
    
        return(std::tuple(tmp_success, tmp_fn_result));
    }
}

DLL_API bool API__Cpp_Python_RPC__Initialize(void)
{
    if(g_Client.Initialized())
    {
        MyEA::File::fError("Initialization can only be call once per load." NEW_LINE
                           "Unload the `.dll` and retry.");
        
        return(false);
    }

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Initialize, std::ref(g_Client),
                                  "C:\\Users\\sebas\\Documents\\MEGAsync\\MyEA\\Python\\run_client.py"));
    
    if(std::get<0>(tmp_results) == false || std::get<1>(tmp_results) == false)
    {
        MyEA::File::fError("An error has been triggered from the `Initialize()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API bool API__Cpp_Python_RPC__Open(void)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(false);
    }

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Open, std::ref(g_Client)));
    
    if(std::get<0>(tmp_results) == false || std::get<1>(tmp_results) == false)
    {
        MyEA::File::fError("An error has been triggered from the `Open()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API bool API__Cpp_Python_RPC__Close(void)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(false);
    }
    else if(Py_Try(&MyEA::RPC::Client::Close, std::ref(g_Client)) == false)
    {
        MyEA::File::fError("An error has been triggered from the `Close()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API T_ API__Cpp_Python_RPC__Predict(void)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Predict, std::ref(g_Client)));
    
    if(std::get<0>(tmp_results) == false)
    {
        MyEA::File::fError("An error has been triggered from the `Predict()` function.");
        
        return(T_EMPTY());
    }

    return(std::get<1>(tmp_results));
}
