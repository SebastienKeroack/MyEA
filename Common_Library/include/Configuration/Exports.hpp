#pragma once

#if defined(COMPILE_DLL_EXPORTS)
    #define DLL_API __declspec(dllexport)
#else
    #define DLL_API __declspec(dllimport)
#endif // COMPILE_DLL_EXPORTS
