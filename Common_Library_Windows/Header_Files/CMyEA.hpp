#pragma once

#include <CObject.hpp>
#include <Arrays/CArrayObject.hpp>
#include <Expert/CExpert.hpp>

namespace MyEA
{
    namespace Common
    {
        class CMyEA : public CObject
        {
            protected:
            CArrayObject p_v_CExpert;

            public:
            CMyEA(void);
            ~CMyEA(void);

            // [     GET      ]
            CArrayObject* Get__v_CExpert(void);
            // ----- GET -----

            const bool Push_Back(CExpert* ptr_CE_received);
        };
    }
}