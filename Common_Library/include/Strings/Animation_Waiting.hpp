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

#pragma once

// Standard.
#include <thread>
#include <atomic>

/// This.
#include <Configuration/Configuration.hpp>

namespace MyEA::String
{
    class Animation_Waiting
    {
        public:
            Animation_Waiting(void);
            ~Animation_Waiting(void);

            void Continue(void);
            void Stop(void);
            void Join(void);
            void Print(void);
            void Print_While(void);
            void Print_While_Async(void);

        private:
            char _switch = 0;

            std::atomic<bool> _continue = true;

            std::thread _thread;
    };
}
