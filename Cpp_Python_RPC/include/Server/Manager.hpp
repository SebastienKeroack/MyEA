#pragma once

// RPC.
#include "rpc/server.h"

class Manager
{
    public:
        Manager(void);
        ~Manager(void);

    private:
        rpc::server *_ptr_server = nullptr;
};
