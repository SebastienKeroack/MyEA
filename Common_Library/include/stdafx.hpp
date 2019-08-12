#pragma once

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#include "targetver.hpp"

#if defined(COMPILE_WINDOWS)
    #define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
    
    #include <windows.h>
#endif

// TODO: reference additional headers your program requires here
#include <Configuration/Configuration.hpp>
