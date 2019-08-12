#pragma once

// Boost.
#include <boost/assert.hpp>

#if defined(COMPILE_FLOAT)
    typedef float ST_;
    
    #if defined(COMPILE_AUTODIFF)
        #include <adept.h>

        typedef adept::afloat T_;

        inline float Cast_T(T_ T_received) { return(T_received.value()); }
    #else
        typedef float T_;

        inline float Cast_T(T_ T_received) { return(T_received); }
    #endif

    typedef int I_;
    #define EPSILON 1.192092896e-07f
#elif defined(COMPILE_DOUBLE)
    typedef double ST_;
    
    #if defined(COMPILE_AUTODIFF)
        #include <adept.h>
        
        typedef adept::adouble T_;

        inline double Cast_T(T_ T_received) { return(T_received.value()); }
    #else
        typedef double T_;

        inline double Cast_T(T_ T_received) { return(T_received); }
    #endif

    typedef long long I_;
    #define EPSILON 2.2204460492503131e-016
#elif defined(COMPILE_LONG_DOUBLE)
    typedef long double T_;
    typedef long long I_;
    #define EPSILON 2.2204460492503131e-016L
#endif

constexpr
T_ operator ""_T(unsigned long long int variable_to_size_t_received)
{
    return(static_cast<T_>(variable_to_size_t_received));
}

constexpr
T_ operator ""_T(long double variable_to_size_t_received)
{
    return(static_cast<T_>(variable_to_size_t_received));
}

constexpr
ST_ operator ""_ST(unsigned long long int variable_to_size_t_received)
{
    return(static_cast<ST_>(variable_to_size_t_received));
}

constexpr
ST_ operator ""_ST(long double variable_to_size_t_received)
{
    return(static_cast<ST_>(variable_to_size_t_received));
}

#if defined(__linux__) || defined(COMPILE_LINUX)
    #include <cstddef>
#endif

constexpr
size_t operator ""_zu(unsigned long long int variable_to_size_t_received)
{
    return(static_cast<size_t>(variable_to_size_t_received));
}

constexpr
size_t KILOBYTE(1024_zu);

// OS
#if defined(__linux__) || defined(COMPILE_LINUX)
    #define TYPE_OS_COMPILE "LINUX"
    
    #include <cstring>
    #include <cmath>
    
    #if defined(COMPILE_COUT)
        #define PRINT_FORMAT printf
        #define PRINT(string) std::cout << string
    #else
        #define PRINT_FORMAT
        #define PRINT(string)
    #endif
    
    #define FSCAN_FORMAT fscanf
#elif defined(__sgi)
    #define TYPE_OS_COMPILE "IRIX"
#elif defined(COMPILE_WINDOWS)
    #define TYPE_OS_COMPILE "COMPILE_WINDOWS"
    
    #if defined(__CUDA_ARCH__)
        #if defined(COMPILE_COUT)
            #define PRINT_FORMAT printf
            #define PRINT(string) printf(string)
        #else
            #define PRINT_FORMAT
            #define PRINT(string)
        #endif
            
        #define FSCAN_FORMAT fscanf
    #else
        #if defined(COMPILE_COUT)
            #define PRINT_FORMAT printf_s
            #define PRINT(string) std::cout << string
        #else
            #define PRINT_FORMAT
            #define PRINT(string)
        #endif
            
        #define FSCAN_FORMAT fscanf_s
    #endif // __CUDA_ARCH__

    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        // https://msdn.microsoft.com/fr-ca/library/x98tx3cf.aspx
        
         // _CRTDBG_MAP_ALLOC
        #if defined(__CUDA_ARCH__) == false
            #define CRTDBG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
            #define new CRTDBG_NEW
            
            #define _CRTDBG_MAP_ALLOC
            #include <cstdlib>
            #include <crtdbg.h>
        #endif // __CUDA_ARCH__
        // |END| _CRTDBG_MAP_ALLOC |END|
        
        //#include <fenv.h>
        //int _feenableexcept_status = feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    #endif  // _DEBUG
#elif defined(sun)
    #define TYPE_OS_COMPILE "SOLARIS"
#elif defined(__APPLE__) && defined(__MACH__) && defined(__ppc__)
    #define TYPE_OS_COMPILE "MacOSX"
#elif defined(macintosh)
    #define TYPE_OS_COMPILE "MacOS"
#elif defined(HPUX10)
    #define TYPE_OS_COMPILE "HPUX10"
#elif defined(HPUX11)
    #define TYPE_OS_COMPILE "HPUX11"
#elif defined(_AIX) || defined(AIX)
    #define TYPE_OS_COMPILE "AIX"
#else
    #define TYPE_OS_COMPILE "UNKNOW"
#endif

// CPU
#if defined(__INTEL__) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
        #define TYPE_CPU_COMPILE "i386"
#elif defined(__POWERPC__) || defined(__PPC__) || defined(__powerpc__) || defined(__ppc__) || defined(_POWER)
        #define TYPE_CPU_COMPILE "PPC"
#elif defined(__m68k__)
    #define TYPE_CPU_COMPILE "68K"
#elif defined(__sgi)
    #define TYPE_CPU_COMPILE "MIPS"
#elif defined(sparc) || defined(__sparc__)
    #define TYPE_CPU_COMPILE "SPARC"
#elif defined(__HP_aCC)
    #define TYPE_CPU_COMPILE "HPPA"
#else
    #define TYPE_CPU_COMPILE "UNKNOWN"
#endif

// COMPILER
#if defined(__GNUG__) || defined(__GNUC__)
    #if __GNUC__ == 2
        #define TYPE_COMPILER_COMPILE "GCC2"
    #elif __GNUC__ == 3
        #define TYPE_COMPILER_COMPILE "GCC3"
    #elif __GNUC__ == 4
        #define TYPE_COMPILER_COMPILE "GCC4"
    #elif __GNUC__ == 5
        #define TYPE_COMPILER_COMPILE "GCC5"
    #elif __GNUC__ == 6
        #define TYPE_COMPILER_COMPILE "GCC6"
    #elif __GNUC__ == 7
        #define TYPE_COMPILER_COMPILE "GCC7"
    #elif __GNUC__ == 8
        #define TYPE_COMPILER_COMPILE "GCC8"
    #else
        #define TYPE_COMPILER_COMPILE "GCC"
    #endif
#elif defined(__BORLANDC__)
    #if __BORLANDC__ < 0x500
        #define TYPE_COMPILER_COMPILE "BCC4"
    #elif __BORLANDC__ < 0x530
        #define TYPE_COMPILER_COMPILE "BCC52"
    #elif __BORLANDC__ < 0x540
        #define TYPE_COMPILER_COMPILE "BCC53"
    #elif __BORLANDC__ < 0x550
        #define TYPE_COMPILER_COMPILE "BCC54"
    #elif __BORLANDC__ < 0x560
        #define TYPE_COMPILER_COMPILE "BCC55"
    #else
        #define TYPE_COMPILER_COMPILE "BCC"
    #endif
#elif defined(__WATCOMC__)
    #define TYPE_COMPILER_COMPILE "WATCOM"
#elif defined(_MIPS_SIM)
    #define TYPE_COMPILER_COMPILE "MIPSCC"
#elif defined(__MWERKS__)
    #define TYPE_COMPILER_COMPILE "MWERKS"
#elif defined(_MSC_VER)
    #if _MSC_VER == 1300
        #define TYPE_COMPILER_COMPILE "VCPP7"
    #elif _MSC_VER == 1200
        #define TYPE_COMPILER_COMPILE "VCPP6"
    #else
        #define TYPE_COMPILER_COMPILE "VCPP"
    #endif
#elif defined(__HP_aCC)
    #define TYPE_COMPILER_COMPILE "ACC"
#elif defined(__IBMCPP__)
    #define TYPE_COMPILER_COMPILE "XLC"
#else
    #define TYPE_COMPILER_COMPILE "UNKNOW"
#endif

// DLL
#if defined(COMPILE_DLL_EXPORTS)
    #define DLL_EXTERNAL extern "C" __declspec(dllexport)
#else
    #define DLL_EXTERNAL extern "C" __declspec(dllimport)
#endif // COMPILE_DLL_EXPORTS

#define DLL_API __stdcall
// |END| DLL |END|

#if defined(NULL) == false
    #define NULL 0
#endif // NULL

#define SAFE_FREE(pointer_received) if(pointer_received) { free(pointer_received); pointer_received = NULL; }

#define SAFE_DELETE(ptr_received) delete(ptr_received); ptr_received = nullptr;

#define SAFE_DELETE_ARRAY(ptr_received) delete[](ptr_received); ptr_received = nullptr;

#if defined(COMPILE_CUDA)
    #if defined(__CUDA_ARCH__) == false
        #define NUMBER_SHARED_MEMORY_BANKS 0u // Number of shared memory banks.
        #define MAXIMUM_THREADS_PER_BLOCK 0u
    #elif __CUDA_ARCH__ < 300
        #define NUMBER_SHARED_MEMORY_BANKS 16u // Number of shared memory banks.
        #define MAXIMUM_THREADS_PER_BLOCK 1024u
    #elif __CUDA_ARCH__ >= 300
        #define NUMBER_SHARED_MEMORY_BANKS 32u // Number of shared memory banks.
        #define MAXIMUM_THREADS_PER_BLOCK 1024u
    #endif
    
    #if defined(__CUDACC__)
        #define __Lch_Bds__(max_threads_per_block_received, min_blocks_per_nultiprocessor_received) __launch_bounds__(max_threads_per_block_received, min_blocks_per_nultiprocessor_received)
    #else
        #define __Lch_Bds__(max_threads_per_block_received, min_blocks_per_nultiprocessor_received)
    #endif
    
    #define FULL_MASK 0xFFFFFFFF
    
    #define USE_PARALLEL false
#endif

void string_cat(char *destination_received,
                size_t const sizeof_received,
                char const *const source_received);
#define STRING_CAT(destination_received, sizeof_received, source_received) string_cat(destination_received, sizeof_received, source_received)

void string_copy(char *destination_received,
                 size_t const sizeof_received,
                 char const *const source_received);
#define STRING_CPY(destination_received, sizeof_received, source_received) string_copy(destination_received, sizeof_received, source_received)

// Type return: The amount of actual physical memory, in bytes.
size_t Get__Total_System_Memory(void);

// Type return: The amount of physical memory currently available, in bytes. This is the amount of physical memory that can be immediately reused without having to write its contents to disk first. It is the sum of the size of the standby, free, and zero lists.
size_t Get__Available_System_Memory(void);

size_t Get__Remaining_Available_System_Memory(long double const reserved_bytes_percent_received, size_t const maximum_reserved_bytes_received);

void PAUSE_TERMINAL(void);

#define PREPROCESSED_CONCAT_(x, y) x##y
#define PREPROCESSED_CONCAT (x, y) PREPROCESSED_CONCAT_(x, y)
    
#define PREPROCESSED_CONCAT_3_(x, y, z) x##y##z
#define PREPROCESSED_CONCAT_3 (x, y, z) PREPROCESSED_CONCAT_3_(x, y, z)
