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

#include <string>

#include <boost/asio/ts/net.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

namespace MyEA
{
namespace Neural_Network
{
    class Model_Manager_Client : public Neural_Network_Manager
    {
        public:
            Model_Manager_Client(bool const is_type_position_long_received, enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicator_received);
            Model_Manager_Client(void);
            ~Model_Manager_Client(void);

            void Deallocate(void);
            void Close(void);
        
            bool Connect(void);
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
}
}