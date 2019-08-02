// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.hpp"

#include <clocale>

namespace MyEA
{
    namespace Network
    {
        BOOL APIENTRY DllMain(HMODULE hModule_received,
                                                DWORD  dwReason_received,
                                                LPVOID lpReserved_received)
        {
            switch(dwReason_received)
            {
                case DLL_PROCESS_ATTACH: std::setlocale(LC_ALL, "en_US.UTF-8"); break;
                case DLL_THREAD_ATTACH:
                case DLL_THREAD_DETACH:
                case DLL_PROCESS_DETACH: break;
            }

            return(TRUE);
        }
    }
}