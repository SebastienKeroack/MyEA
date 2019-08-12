#pragma once

// Standard.
#include <string>
#include <map>

namespace MyEA::UI
{
    enum ENUM_TYPE_DIALOG_BOX : unsigned int
    {
        ACCEPT = 0u,
        OK     = 1u,
        LENGTH = 2u
    };

    static
    std::map<enum ENUM_TYPE_DIALOG_BOX, std::string> ENUM_TYPE_DIALOG_BOX_NAMES = {
            {ACCEPT, "Accept"},
            {OK,     "OK"},
            {LENGTH, "Length"}
                                                                                  };
}