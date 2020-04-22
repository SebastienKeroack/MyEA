#include "pch.hpp"

#include "Cpp_With_Python.hpp"
#include "Client.hpp"

Client g_client;

std::string wchar_t_as_string(wchar_t const *const value)
{
    std::wstring const value_as_wstring(value);

    return(boost::locale::conv::utf_to_utf<char>(value_as_wstring));
}

DLL_API bool API__Connect(wchar_t const *const script, wchar_t const *const host)
{
    if(g_client.Initialized() == false)
    {
        g_client.Initialize(wchar_t_as_string(script), wchar_t_as_string(host));
    }

    if(g_client.Is_Connected())
    {
        return(true);
    }

    return(g_client.Open());
}

DLL_API bool API__Is_Connected(void)
{
    if(g_client.Initialized() == false) { return(false); }

    return(g_client.Is_Connected());
}

DLL_API unsigned int API__CppCurrentTime(void)
{
    if(g_client.Initialized() == false) { return(0u); }
    
    return(g_client.CppCurrentTime());
}

DLL_API int API__Action(void)
{
    if(g_client.Initialized() == false) { return(1); }
    
    return(g_client.Action());
}

DLL_API void API__Disconnect(void)
{
    g_client.Close();
}