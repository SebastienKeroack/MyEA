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
