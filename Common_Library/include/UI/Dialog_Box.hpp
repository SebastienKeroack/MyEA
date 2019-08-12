#pragma once

// Standard.
#include <string>

// This.
#include <Configuration/Configuration.hpp>
#include <Enums/Enum_Type_Dialog_Box.hpp>

namespace MyEA::UI
{
    bool Dialog_Box__Accept(std::string const &ref_text_received, std::string const &ref_title_received);

    bool Dialog_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received);

    bool Dialog_Box(ENUM_TYPE_DIALOG_BOX const type_received,
                    std::string const &ref_text_received,
                    std::string const &ref_title_received);
}

#define DEBUG_BOX(text_received) MyEA::UI::Dialog_Box(MyEA::UI::ENUM_TYPE_DIALOG_BOX::OK, text_received, "DEBUG");