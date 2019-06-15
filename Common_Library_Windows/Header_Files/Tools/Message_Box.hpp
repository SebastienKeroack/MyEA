#pragma once

#include <string>

#include <Tools/Configuration.hpp>

namespace MyEA
{
    namespace Common
    {
        bool Message_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received);

        bool Message_Box__YESNO(std::string const &ref_text_received, std::string const &ref_title_received);
    }
}

#define DEBUG_BOX(text_received) MyEA::Common::Message_Box__OK(text_received, "DEBUG");