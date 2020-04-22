#include "stdafx.hpp"

#include <Capturing/Shutdown/Shutdown.hpp>
#include <UI/Dialog_Box.hpp>
#include <Strings/String.hpp>
#include <Files/File.hpp>

namespace MyEA
{
    namespace Common
    {
        DLL_API bool API__Common__Shutdown_Block__Create(bool const use_ctrl_handler_received,
                                                                                                                     wchar_t const *const ptr_window_name_received,
                                                                                                                     wchar_t const *const ptr_class_name_received)
        {
            if(ptr_global_Shutdown_Block == nullptr)
            {
                std::wstring const tmp_window_name(ptr_window_name_received),
                                           tmp_class_name(ptr_class_name_received);

                ptr_global_Shutdown_Block = new class MyEA::Capturing::Shutdown(std::string(tmp_window_name.begin(), tmp_window_name.end()), std::string(tmp_class_name.begin(), tmp_class_name.end()));
                
                if(ptr_global_Shutdown_Block != nullptr && ptr_global_Shutdown_Block->Create_Shutdown_Block(use_ctrl_handler_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Create_Shutdown_Block(%u)\" function." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             use_ctrl_handler_received ? 1u : 0u);

                    return(false);
                }

                return(true);
            }
            else { return(false); }
        }
        
        DLL_API bool API__Common__Shutdown_Block__Remove(void)
        {
            if(ptr_global_Shutdown_Block != nullptr)
            {
                SAFE_DELETE(ptr_global_Shutdown_Block);

                return(true);
            }
            else { return(false); }
        }
        
        DLL_API bool API__Common__Shutdown_Block__Peek_Message(void)
        {
            if(ptr_global_Shutdown_Block != nullptr)
            {
                if(ptr_global_Shutdown_Block->Peak_Message() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Peak_Message()\" function." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__);

                    return(false);
                }

                return(true);
            }
            else { return(false); }
        }

        DLL_API int API__Common__Shutdown_Block__Get__On_Shutdown(void)
        {
            if(ptr_global_Shutdown_Block != nullptr)
            { return(ptr_global_Shutdown_Block->Get__On_Shutdown() ? 1 : 0); }
            else
            { return(0); }
        }

        DLL_API bool API__Common__Path_Exist(wchar_t const *const ptr_path_received)
        {
            std::wstring const tmp_wstring(ptr_path_received);

            return(MyEA::File::Path_Exist(std::string(tmp_wstring.begin(), tmp_wstring.end())));
        }

        DLL_API bool API__Common__Message_Box__OK(wchar_t const *const ptr_ws_text_received, wchar_t const *const ptr_ws_title_received)
        {
            std::wstring const tmp_ws_text_received(ptr_ws_text_received),
                                       tmp_ws_title_received(ptr_ws_title_received);

            return(MyEA::Common::Message_Box__OK(std::string(tmp_ws_text_received.begin(), tmp_ws_text_received.end()), std::string(tmp_ws_title_received.begin(), tmp_ws_title_received.end())));
        }

        DLL_API bool API__Common__Message_Box__YESNO(wchar_t const *const ptr_ws_text_received, wchar_t const *const ptr_ws_title_received)
        {
            std::wstring const tmp_ws_text_received(ptr_ws_text_received),
                                       tmp_ws_title_received(ptr_ws_title_received);

            return(MyEA::Common::Message_Box__YESNO(std::string(tmp_ws_text_received.begin(), tmp_ws_text_received.end()), std::string(tmp_ws_title_received.begin(), tmp_ws_title_received.end())));
        }

        DLL_API bool API__Common__Write_File(wchar_t const *const path_received,
                                                                                              wchar_t const *const log_received,
                                                                                              int const mode_received)
        {
            std::wstring const tmp_wstring_path(path_received),
                                       tmp_wstring_log(log_received);

            return(MyEA::File::Write_File(std::string(tmp_wstring_path.begin(), tmp_wstring_path.end()),
                                                       std::string(tmp_wstring_log.begin(), tmp_wstring_log.end()),
                                                       mode_received));
        }
    }
}