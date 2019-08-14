#include "pch.hpp"

// Standard.
#include <iostream>
#include <sys/ioctl.h>
#include <termios.h>

// This.
#include <Capturing/Keyboard/Linux/Keyboard.hpp>

namespace MyEA::Capturing
{
    Keyboard::Keyboard(void)
    {
        static bool _termios_initialized = false;

        // Use termios to turn off line buffering.
        if(_termios_initialized == false)
        {
            termios tmp_termios;

            tcgetattr(0, &tmp_termios);

            tmp_termios.c_lflag &= ~ICANON;

            tcsetattr(0,
                      TCSANOW,
                      &tmp_termios);

            setbuf(stdin, NULL);

            _termios_initialized = true;
        }
    }

    int Keyboard::_KBHIT(void)
    {
        int tmp_bytes_waiting;

        ioctl(0,
              FIONREAD,
              &tmp_bytes_waiting);

        return(tmp_bytes_waiting);
    }

    void Keyboard::Clear_Keys_Pressed(void)
    {
        this->_map_chars.clear();
    }

    void Keyboard::Collect_Keys_Pressed(void)
    {
        int tmp_number_bytes;

        if((tmp_number_bytes = this->_KBHIT()) != 0)
        {
            char *tmp_cin_buffers = new char[tmp_number_bytes];

            std::cin.read(tmp_cin_buffers, tmp_number_bytes);

            this->_map_chars = std::string(tmp_cin_buffers, tmp_number_bytes);

            delete[](tmp_cin_buffers);
        }
    }

    bool Keyboard::Trigger_Key(char const char_received)
    {
        return(this->_map_chars.find(char_received) !=std::string::npos);
    }
}