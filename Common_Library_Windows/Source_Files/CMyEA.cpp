#include "stdafx.hpp"

#include <CMyEA.hpp>

namespace MyEA
{
    namespace Common
    {
        CMyEA::CMyEA(void)
        {
        }

        CArrayObject* CMyEA::Get__v_CExpert(void) { return(&this->p_v_CExpert); }

        const bool CMyEA::Push_Back(CExpert* ptr_CE_received) { return(this->Get__v_CExpert()->Push_Back(ptr_CE_received)); }

        CMyEA::~CMyEA(void)
        {
        }
    }
}
