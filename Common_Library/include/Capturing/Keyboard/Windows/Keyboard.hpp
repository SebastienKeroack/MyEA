#pragma once

namespace MyEA::Capturing
{
    class Keyboard
    {
        public:
            Keyboard(void);

            bool Trigger_Key(short const key_code);
    };
}