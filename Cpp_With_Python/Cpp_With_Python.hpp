#pragma once

DLL_API bool API__Connect(wchar_t const *const script, wchar_t const *const host);

DLL_API bool API__Is_Connected(void);

DLL_API unsigned int API__CppCurrentTime(void);

DLL_API int API__Action(void);

DLL_API void API__Disconnect(void);