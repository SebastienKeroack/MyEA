// Std.
#include <iostream>

// THIS
#include <Client/Manager.hpp>

Manager::Manager(void)
{
    this->_ptr_client = new rpc::client("127.0.0.1", 9000);

    this->_ptr_client->call("foo");
}

Manager::~Manager(void)
{
    if(this->_ptr_client != nullptr)
    {
        delete(this->_ptr_client);
        this->_ptr_client = nullptr;
    }
}
