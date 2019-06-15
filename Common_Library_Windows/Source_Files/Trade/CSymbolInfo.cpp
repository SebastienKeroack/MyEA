#include "stdafx.hpp"

#include <Trade/CSymbolInfo.hpp>

namespace MyEA
{
    namespace Common
    {
        CSymbolInfo::CSymbolInfo(void) : _name("XXXYYY")
        {
        }

        CSymbolInfo::~CSymbolInfo(void)
        {
        }

        const std::string CSymbolInfo::Get__Name(void) { return(this->_name); }

        const bool CSymbolInfo::Set__Name(const std::string name_received)
        {
            this->_name = name_received;

            return(true);
        }
    }
}
