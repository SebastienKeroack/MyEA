#pragma once

namespace MyEA::RPC
{
    class Client
    {
        public:
            Client(void);
            
            ~Client(void);

            bool Initialized(void) const;

            bool Initialize(void);

            void Call(void);
    };
}