#include <pch.hpp>

// Boost.
#include <boost/python.hpp>

// This.
#include <Client/Client.hpp>

class MyEA::RPC::Client *g_Manager = nullptr;

DLL_API void API__Cpp_Python_RPC__Dialog_Box(void)
{
    g_Manager = new MyEA::RPC::Client;

    g_Manager->foo();

    delete(g_Manager);
}

BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD ul_reason_for_call,
                      LPVOID lpReserved)
{
    switch(ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH: Py_Initialize(); break;
        case DLL_THREAD_ATTACH :
        case DLL_THREAD_DETACH :
        case DLL_PROCESS_DETACH: break;
    }

    return(TRUE);
}