#include "stdafx.hpp"

#include <UI/Message_Box.hpp>

namespace MyEA::UI
{
#if defined(COMPILE_LINUX)
    std::string Get__Dialog_Command(void)
    {
        if(::system(NULL))
        {
            if(     ::system("which gdialog") == 0) { return("gdialog"); }
            else if(::system("which kdialog") == 0) { return("kdialog"); }
        }

        return("");
    }

    bool Linux__Message_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        std::string const tmp_dialog_command(Get__Dialog_Command());

        if(tmp_dialog_command.empty() == false)
        {
            std::string const tmp_command(tmp_dialog_command + " --title \""+ ref_title_received + "\" --msgbox \"" + ref_text_received + "\"");

            int const tmp_result(::system(tmp_command.c_str()));

            if(tmp_result == 0) { return(true); }
        }

        return(false);
    }

    bool Linux__Message_Box__YESNO(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        std::string const tmp_dialog_command(Get__Dialog_Command());

        if(tmp_dialog_command.empty() == false)
        {
            std::string const tmp_command(tmp_dialog_command + " --title \""+ ref_title_received + "\" --yesno \"" + ref_text_received + "\"");

            int const tmp_result(::system(tmp_command.c_str()));

            if(     tmp_result == 0) { return(true ); }
            else if(tmp_result == 1) { return(false); }
        }

        return(false);
    }
#endif

    bool Message_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received)
    {
    #if defined(COMPILE_WINDOWS)
        int const tmp_message_box_ID = MessageBox(NULL,
                                                  ref_text_received.c_str(),
                                                  ref_title_received.c_str(),
                                                  MB_ICONINFORMATION | MB_OK);

        switch(tmp_message_box_ID)
        {
            case IDOK: return(true);
            default: return(true);
        }
    #elif defined(COMPILE_LINUX)
        return(Linux__Message_Box__OK(ref_text_received, ref_title_received));
    #endif
    }

    bool Message_Box__YESNO(std::string const &ref_text_received, std::string const &ref_title_received)
    {
    #if defined(COMPILE_WINDOWS)
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
    #elif defined(COMPILE_LINUX)
        return(Linux__Message_Box__YESNO(ref_text_received, ref_title_received));
    #endif
    }
}
