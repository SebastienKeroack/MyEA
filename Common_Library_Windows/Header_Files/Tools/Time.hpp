#pragma once

#include <string>

namespace MyEA
{
    namespace Time
    {
        void Sleep(unsigned int const milliseconds_received);

        std::string Get__DateTimeFull(void);
        std::string Get__DateTimeStandard(void);
        std::string Get__DateTimeMinimal(void);
    }
}