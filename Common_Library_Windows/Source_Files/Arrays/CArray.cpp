#include "stdafx.hpp"

#include <Arrays/CArray.hpp>

namespace MyEA
{
    namespace Common
    {
        CArray::CArray(void) : p_data_total(0),
                                         p_data_max(0)
        {
        }

        CArray::~CArray(void)
        {
        }
    }
}
