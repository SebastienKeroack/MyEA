#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_INITPHASE
        {
            INITPHASE_AWAKE = 0,
            INITPHASE_TUNING = 1,
            INITPHASE_VALIDATION = 2,
            INITPHASE_COMPLETE = 3
        };

        static std::map<enum ENUM_INITPHASE, std::string> ENUM_INITPHASE_NAMES = {{INITPHASE_AWAKE, "INITPHASE_AWAKE"},
                                                                                                                                     {INITPHASE_TUNING, "INITPHASE_TUNING"},
                                                                                                                                     {INITPHASE_VALIDATION, "INITPHASE_VALIDATION"},
                                                                                                                                     {INITPHASE_COMPLETE, "INITPHASE_COMPLETE"}};
    }
}