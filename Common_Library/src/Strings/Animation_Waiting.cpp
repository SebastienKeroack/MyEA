/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pch.hpp"

// This.
#include <Strings/Animation_Waiting.hpp>

// Standard.
#include <iostream>

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
