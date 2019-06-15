#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        struct TIME_FRAMES
        {
            unsigned int TIME_FRAMES_PERIOD_CURRENT;
            unsigned int TIME_FRAMES_PERIOD_M1;
            unsigned int TIME_FRAMES_PERIOD_M2;
            unsigned int TIME_FRAMES_PERIOD_M3;
            unsigned int TIME_FRAMES_PERIOD_M4;
            unsigned int TIME_FRAMES_PERIOD_M5;
            unsigned int TIME_FRAMES_PERIOD_M6;
            unsigned int TIME_FRAMES_PERIOD_M10;
            unsigned int TIME_FRAMES_PERIOD_M12;
            unsigned int TIME_FRAMES_PERIOD_M15;
            unsigned int TIME_FRAMES_PERIOD_M20;
            unsigned int TIME_FRAMES_PERIOD_M30;
            unsigned int TIME_FRAMES_PERIOD_H1;
            unsigned int TIME_FRAMES_PERIOD_H2;
            unsigned int TIME_FRAMES_PERIOD_H3;
            unsigned int TIME_FRAMES_PERIOD_H4;
            unsigned int TIME_FRAMES_PERIOD_H6;
            unsigned int TIME_FRAMES_PERIOD_H8;
            unsigned int TIME_FRAMES_PERIOD_H12;
            unsigned int TIME_FRAMES_PERIOD_D1;
            unsigned int TIME_FRAMES_PERIOD_W1;
            unsigned int TIME_FRAMES_PERIOD_MN1;
        };

        enum ENUM_TIME_FRAMES : unsigned int
        {
            TIME_FRAMES_PERIOD_NONE = 0u,
            TIME_FRAMES_PERIOD_M1 = 60u,
            TIME_FRAMES_PERIOD_M2 = 120u,
            TIME_FRAMES_PERIOD_M3 = 180u,
            TIME_FRAMES_PERIOD_M4 = 240u,
            TIME_FRAMES_PERIOD_M5 = 300u,
            TIME_FRAMES_PERIOD_M6 = 360u,
            TIME_FRAMES_PERIOD_M10 = 600u,
            TIME_FRAMES_PERIOD_M12 = 720u,
            TIME_FRAMES_PERIOD_M15 = 900u,
            TIME_FRAMES_PERIOD_M20 = 1'200u,
            TIME_FRAMES_PERIOD_M30 = 1'800u,
            TIME_FRAMES_PERIOD_H1 = 3'600u,
            TIME_FRAMES_PERIOD_H2 = 7'200u,
            TIME_FRAMES_PERIOD_H3 = 10'800u,
            TIME_FRAMES_PERIOD_H4 = 14'400u,
            TIME_FRAMES_PERIOD_H6 = 21'600u,
            TIME_FRAMES_PERIOD_H8 = 28'800u,
            TIME_FRAMES_PERIOD_H12 = 43'200u,
            TIME_FRAMES_PERIOD_D1 = 86'400u,
            TIME_FRAMES_PERIOD_W1 = 604'800u,
            TIME_FRAMES_PERIOD_MN1 = 2'505'600u
        };

        static std::map<enum ENUM_TIME_FRAMES, std::string> ENUM_TIME_FRAMES_NAMES = {{TIME_FRAMES_PERIOD_NONE, "NONE"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M1, "M1"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M2, "M2"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M3, "M3"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M4, "M4"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M5, "M5"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M6, "M6"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M10, "M10"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M12, "M12"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M15, "M15"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M20, "M20"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_M30, "M30"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H1, "H1"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H2, "H2"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H3, "H3"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H4, "H4"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H6, "H6"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H8, "H8"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_H12, "H12"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_D1, "D1"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_W1, "W1"},
                                                                                                                                                   {TIME_FRAMES_PERIOD_MN1, "MN1"}};
    }
}