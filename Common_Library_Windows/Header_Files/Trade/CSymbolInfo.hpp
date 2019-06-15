#pragma once

#include <string>

#include <CObject.hpp>

namespace MyEA
{
    namespace Common
    {
        class CSymbolInfo : public CObject
        {
            public:
                CSymbolInfo(void);
                ~CSymbolInfo(void);

                const std::string Get__Name(void);

                const bool Set__Name(const std::string name_received);

            private:
                std::string _name;
        };
    }
}