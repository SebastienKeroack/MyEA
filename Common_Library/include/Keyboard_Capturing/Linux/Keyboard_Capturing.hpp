#pragma once

#include <string>

class Key_Logger
{
    public:
        Key_Logger(void);

        void Clear_Keys_Pressed(void);
        void Collect_Keys_Pressed(void);

        bool Trigger_Key(char const char_received);

    private:
        int _KBHIT(void);

        std::string _map_chars;
};