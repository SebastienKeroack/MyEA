// Std.
#include <iostream>

// THIS
#include <Server/Manager.hpp>

void foo(void) { std::cout << "foo was called!" << std::endl; }

Manager::Manager(void)
{
    this->_ptr_server = new rpc::server(9000);

    this->_ptr_server->bind("foo", &foo);

    this->_ptr_server->async_run();
}

Manager::~Manager(void)
{
    if(this->_ptr_server != nullptr)
    {
        this->_ptr_server->stop();

        delete(this->_ptr_server);
        this->_ptr_server = nullptr;
    }
}

// https://github.com/rpclib/rpclib/issues/100