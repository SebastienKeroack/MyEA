#include "stdafx.hpp"

// This.
#include <UI/Dialog_Box.hpp>

namespace MyEA::UI
{
    bool Dialog_Box__Accept(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        int const tmp_message_box_ID = MessageBox(NULL,
                                                  ref_text_received.c_str(),
                                                  ref_title_received.c_str(),
                                                  MB_ICONINFORMATION | MB_YESNO);

        switch(tmp_message_box_ID)
        {
            case IDYES: return(true );
            case IDNO : return(false);
            default: return(false);
        }
    }

    bool Dialog_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        int const tmp_message_box_ID = MessageBox(NULL,
                                                  ref_text_received.c_str(),
                                                  ref_title_received.c_str(),
                                                  MB_ICONINFORMATION | MB_OK);

        switch(tmp_message_box_ID)
        {
            case IDOK: return(true);
            default: return(true);
        }
    }
}
