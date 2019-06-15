#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_EVENTSIGNAL_WAIT
        {
            EVENTSIGNAL_WAIt_NONE = 0,
            EVENTSIGNAL_WAITCANDLESTICK = 1,
            EVENTSIGNAL_WAITM1 = 2,
            EVENTSIGNAL_WAITM5 = 3,
            EVENTSIGNAL_WAITM15 = 4,
            EVENTSIGNAL_WAITH1 = 5,
            EVENTSIGNAL_WAITH4 = 6,
            EVENTSIGNAL_WAITD1 = 7,
            EVENTSIGNAL_WAITW1 = 8,
            EVENTSIGNAL_WAITMN1 = 9
        };

        static std::map<enum ENUM_EVENTSIGNAL_WAIT, std::string> ENUM_EVENTSIGNAL_WAIt_NAMES = {{EVENTSIGNAL_WAIt_NONE, "EVENTSIGNAL_WAIt_NONE"},
                                                                                                                                                                    {EVENTSIGNAL_WAITCANDLESTICK, "EVENTSIGNAL_WAITCANDLESTICK"},
                                                                                                                                                                    {EVENTSIGNAL_WAITM1, "EVENTSIGNAL_WAITM1"},
                                                                                                                                                                    {EVENTSIGNAL_WAITM5, "EVENTSIGNAL_WAITM5"},
                                                                                                                                                                    {EVENTSIGNAL_WAITM15, "EVENTSIGNAL_WAITM15"},
                                                                                                                                                                    {EVENTSIGNAL_WAITH1, "EVENTSIGNAL_WAITH1"},
                                                                                                                                                                    {EVENTSIGNAL_WAITH4, "EVENTSIGNAL_WAITH4"},
                                                                                                                                                                    {EVENTSIGNAL_WAITD1, "EVENTSIGNAL_WAITD1"},
                                                                                                                                                                    {EVENTSIGNAL_WAITW1, "EVENTSIGNAL_WAITW1"},
                                                                                                                                                                    {EVENTSIGNAL_WAITMN1, "EVENTSIGNAL_WAITMN1"}};

    }
}