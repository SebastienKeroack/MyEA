#pragma once

// RPC.
#include "rpc/client.h"

class Manager
{
    public:
        Manager(void);
        ~Manager(void);

    private:
        rpc::client *_ptr_client = nullptr;
};
