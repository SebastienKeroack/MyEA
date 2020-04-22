// stdafx.cpp : source file that includes just the standard includes
// CONSOLE.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.hpp"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

#if defined(COMPILE_AUTODIFF)
/* Need to update the sources files of Adept, to contain the following functions in the specified file and class:
    UnaryOperation.h:
        struct UnaryOperation:
            operator size_t() const { return(static_cast<size_t>(this->cast())); }
    BinaryOperation.h:
        struct BinaryOperation:
            operator size_t() const { return(static_cast<size_t>(this->cast())); }
    Active.h:
        class Active:
            operator size_t() const { return(static_cast<size_t>(val_)); } */

    // Initialize stack.
    adept::Stack global_Adept_Stack;
#endif
