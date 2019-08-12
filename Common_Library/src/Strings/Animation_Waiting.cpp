#include "stdafx.hpp"

#include <string>
#include <iostream>

#include <Strings/Animation_Waiting.hpp>

namespace MyEA::String
{
    Animation_Waiting::Animation_Waiting(void) { }

    Animation_Waiting::~Animation_Waiting(void) { }

    void Animation_Waiting::Continue(void)
    {
        this->_continue.store(true);
    }

    void Animation_Waiting::Stop(void)
    {
        this->_continue.store(false);
    }

    void Animation_Waiting::Join(void)
    {
        if(this->_thread.joinable())
        {
            this->Stop();

            this->_thread.join();
        }
    }

    void Animation_Waiting::Print(void)
    {
        // Print corresponding pattern.
        switch(this->_switch)
        {
            case 0: std::cout << '[' << '/'  << ']'; break;
            case 1: std::cout << '[' << '-'  << ']'; break;
            case 2: std::cout << '[' << '\\' << ']'; break;
            default: std::cout << "ERR"; break;
        }

        // If overflow set it to zero.
        if(++this->_switch == 3) { this->_switch = 0; }
    }

    void Animation_Waiting::Print_While(void)
    {
        while(this->_continue.load())
        {
            // Print corresponding pattern.
            this->Print();

            // Sleep for smooth animation.
            std::this_thread::sleep_for(std::chrono::milliseconds(125));

            // Cursor go back 3 cases.
            std::cout << std::string(3, '\b');
        }

        // Clear waiting characters.
        std::cout << std::string(3, ' ');
    }

    void Animation_Waiting::Print_While_Async(void)
    {
        if(this->_thread.joinable() == false)
        {
            this->Continue();

            this->_thread = std::thread(&Animation_Waiting::Print_While, this);
        }
    }
}
