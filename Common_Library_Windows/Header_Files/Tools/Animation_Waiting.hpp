#pragma once

#include <thread>
#include <atomic>

#include <Tools/Configuration.hpp>

namespace MyEA
{
    namespace Animation
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
                char _bytes = 0;

                std::atomic<bool> _continue = true;

                std::thread _thread;
        };
    }
}
