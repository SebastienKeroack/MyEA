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