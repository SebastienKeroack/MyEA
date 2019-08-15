#include "pch.hpp"

// This.
#include <API.hpp>
#include <Client/Client.hpp>

class MyEA::RPC::Client *g_Manager = nullptr;

DLL_API void API__Cpp_Python_RPC__Dialog_Box(void)
{
    g_Manager = new MyEA::RPC::Client;

    g_Manager->foo();

    delete(g_Manager);
}
