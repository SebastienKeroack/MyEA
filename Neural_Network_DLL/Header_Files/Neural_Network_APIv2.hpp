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

#include <Configuration/Configuration.hpp>

#include <vector>
#include <thread>

#include <Model_Manager_Client.hpp>

namespace MyEA::Neural_Network
{
    class Client
    {
        public:
            Client(void);
            ~Client(void);

            void Deallocate(void);
            void Close(void);
                
            bool Connect(void);
            bool Connected(void) const;
            bool Send(std::string const buffer_received) const;
                
            int Done(void) const;
            int Get__Port(void) const;

            std::string Receive(void) const;

        private:
            bool _Port_In_Use(unsigned short const port_received);
                
            unsigned short _Find_Opened_Ports(void);
            unsigned short _port = 0;

            boost::asio::io_context _io_context;
                
            boost::asio::ip::tcp::resolver *_ptr_resolver = nullptr;

            boost::asio::ip::tcp::socket *_ptr_socket = nullptr;
    };
        
    extern class Client *g_Client;

    DLL_API bool API__Neural_Network__Is_Loaded(void);
        
    DLL_API bool API__Neural_Network__Allocate(void);

    DLL_API bool API__Neural_Network__Initialize(void);
        
    DLL_API bool API__Neural_Network__Deinitialize(void);
    
    DLL_API float API__Neural_Network__Get__Loss(unsigned int const type_neural_network_use_received, unsigned int const type_loss_received);
    
    DLL_API T_ API__Neural_Network__Forward_Pass(T_ *const ptr_array_inputs_received);
}