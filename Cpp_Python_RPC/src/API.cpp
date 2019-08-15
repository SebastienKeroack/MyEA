#include "pch.hpp"

// This.
#include <API.hpp>
#include <Client/Client.hpp>

// Common_Library.
#include <Strings/String.hpp>

class MyEA::RPC::Client g_Client;

DLL_API bool API__Cpp_Python_RPC__Initialize(void)
{
    if(g_Client.Initialize() == false)
    {
        MyEA::String::Error("Initialization can only be call once per load." NEW_LINE
                            "Unload the `.dll` and retry.");
        
        return(false);
    }
    
    return(true);
}
