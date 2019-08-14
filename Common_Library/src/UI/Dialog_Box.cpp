#include "pch.hpp"

// This.
#include <Strings/String.hpp>
#include <UI/Dialog_Box.hpp>

namespace MyEA::UI
{
    bool Dialog_Box(ENUM_TYPE_DIALOG_BOX const type_received,
                    std::string const &ref_text_received,
                    std::string const &ref_title_received)
    {
        switch(type_received)
        {
            case ENUM_TYPE_DIALOG_BOX::ACCEPT: return(Dialog_Box__Accept(ref_text_received, ref_title_received));
            case ENUM_TYPE_DIALOG_BOX::OK    : return(Dialog_Box__OK    (ref_text_received, ref_title_received));
            default:
                MyEA::String::Error("The `%s` dialog box type is not supported in the switch.", ENUM_TYPE_DIALOG_BOX_NAMES[type_received].c_str());
                    return(false);
        }
    }
}
