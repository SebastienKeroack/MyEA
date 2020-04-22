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

#include "stdafx.hpp"

#include <limits>

#include <Model_Manager_Client.hpp>

namespace MyEA
{
namespace Neural_Network
{
    Model_Manager_Client::Model_Manager_Client(bool const is_type_position_long_received, enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicator_received) : Neural_Network_Manager(is_type_position_long_received, type_indicator_received) { }

    Model_Manager_Client::Model_Manager_Client(void) : Neural_Network_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE) { }

    Model_Manager_Client::~Model_Manager_Client(void) { this->Deallocate(); }
    
    void Model_Manager_Client::Deallocate(void)
    {
        delete(this->_ptr_resolver);

        if(this->_ptr_socket != nullptr)
        {
            this->_ptr_socket->close();

            delete(this->_ptr_socket);
        }
    }
    
    void Model_Manager_Client::Close(void) { this->Deallocate(); }
    
    bool Model_Manager_Client::Connect(void)
    {
        if(this->_ptr_resolver != nullptr
            ||
            this->_ptr_socket != nullptr) { return(false); }
            
        unsigned short tmp_port(this->_Find_Opened_Ports());

        if(tmp_port == (std::numeric_limits<unsigned short>::max)()) { return(false); }

        this->_ptr_resolver = new boost::asio::ip::tcp::resolver(this->_io_context);

        this->_ptr_socket = new boost::asio::ip::tcp::socket(this->_io_context);

        boost::asio::connect(*this->_ptr_socket, this->_ptr_resolver->resolve("localhost", std::to_string(tmp_port)));

        this->_port = tmp_port;

        return(true);
    }

    bool Model_Manager_Client::Send(std::string const buffer_received) const
    {
        if(this->_ptr_socket == nullptr) { return(false); }

        this->_ptr_socket->write_some(boost::asio::buffer(buffer_received));

        return(true);
    }
    
    bool Model_Manager_Client::_Port_In_Use(unsigned short const port_received)
    {
        boost::asio::io_context tmp_io_context;
    
        boost::asio::ip::tcp::acceptor tmp_acceptor(tmp_io_context);

        boost::system::error_code tmp_error_code;

        tmp_acceptor.open(boost::asio::ip::tcp::v4(), tmp_error_code);

        if(tmp_error_code) { return(false); }

        tmp_acceptor.bind({boost::asio::ip::tcp::v4(), port_received}, tmp_error_code);

        return(tmp_error_code == boost::asio::error::address_in_use);
    }
    
    int Model_Manager_Client::Done(void) const
    {
        if(this->_ptr_socket == nullptr) { return(-1); }
            
        std::string tmp_buffer("");
            
        boost::system::error_code tmp_error_code;
            
        size_t const tmp_bytes(boost::asio::read(*this->_ptr_socket,
                                                                    boost::asio::dynamic_buffer(tmp_buffer),
                                                                    [&tmp_buffer](auto ec, auto n) -> std::size_t
                                                                    {
                                                                        if(ec
                                                                            ||
                                                                            (
                                                                            tmp_buffer.size() >= 1
                                                                            &&
                                                                            tmp_buffer.compare(tmp_buffer.size() - 1,
                                                                                                            1,
                                                                                                            "$") == 0
                                                                            )) { return(0); }
                                                                            
                                                                        return(1);
                                                                    },
                                                                    tmp_error_code));

        if(tmp_error_code) { return(-1); }

        int tmp_return_code;

        try
        {
            tmp_return_code = std::stoi(tmp_buffer.substr(0, tmp_buffer.size() - 1));
        }
        catch(...)
        {
            tmp_return_code = -1;
        }

        return(tmp_return_code);
    }

    int Model_Manager_Client::Get__Port(void) const { return(this->_port); }

    unsigned short Model_Manager_Client::_Find_Opened_Ports(void)
    {
        for(unsigned short tmp_port_index(9061); tmp_port_index < 9079; ++tmp_port_index)
        {
            if(this->_Port_In_Use(tmp_port_index)) { return(tmp_port_index); }
        }

        return((std::numeric_limits<unsigned short>::max)());
    }
    
    std::string Model_Manager_Client::Receive(void) const
    {
        if(this->_ptr_socket == nullptr) { return(""); }
            
        std::string tmp_buffer("");
            
        boost::system::error_code tmp_error_code;

        size_t const tmp_bytes(boost::asio::read(*this->_ptr_socket,
                                                                    boost::asio::dynamic_buffer(tmp_buffer),
                                                                    [&tmp_buffer](auto ec, auto n) -> std::size_t
                                                                    {
                                                                        if(ec
                                                                            ||
                                                                            (
                                                                            tmp_buffer.size() >= 1
                                                                            &&
                                                                            tmp_buffer.compare(tmp_buffer.size() - 1,
                                                                                                            1,
                                                                                                            "$") == 0
                                                                            )) { return(0); }
                                                                            
                                                                        return(1);
                                                                    },
                                                                    tmp_error_code));

        if(tmp_error_code) { return(""); }

        return(tmp_buffer.substr(0, tmp_buffer.size() - 1));
    }
}
}