#include "stdafx.hpp"

#include <Enums/Enum_Time_Frames.hpp>

#include <Tools/Message_Box.hpp>

#include <Neural_Network_APIv2.hpp>

namespace MyEA::Neural_Network
{
    enum QUERY_CODE : unsigned char
    {
        QUERY_PREDICT = '0',
        QUERY_MODEL_HISTORY = '1'
    };

    Client::Client(void) { }

    Client::~Client(void) { this->Deallocate(); }
    
    void Client::Deallocate(void)
    {
        SAFE_DELETE(this->_ptr_resolver);

        if(this->_ptr_socket != nullptr)
        {
            this->_ptr_socket->close();

            SAFE_DELETE(this->_ptr_socket);
        }
    }
    
    void Client::Close(void) { this->Deallocate(); }
    
    bool Client::Connect(void)
    {
        if(this->_ptr_resolver != nullptr
            ||
            this->_ptr_socket != nullptr) { return(false); }
            
        unsigned short const tmp_port(this->_Find_Opened_Ports());

        if(tmp_port == 0) { return(false); }

        this->_ptr_resolver = new boost::asio::ip::tcp::resolver(this->_io_context);

        this->_ptr_socket = new boost::asio::ip::tcp::socket(this->_io_context);

        boost::asio::connect(*this->_ptr_socket, this->_ptr_resolver->resolve("localhost", std::to_string(tmp_port)));

        this->_port = tmp_port;

        return(true);
    }
        
    bool Client::Connected(void) const { return(this->_port != 0); }

    bool Client::Send(std::string const buffer_received) const
    {
        if(this->_ptr_socket == nullptr) { return(false); }

        this->_ptr_socket->write_some(boost::asio::buffer(buffer_received));

        return(true);
    }
    
    bool Client::_Port_In_Use(unsigned short const port_received)
    {
        boost::asio::io_context tmp_io_context;
    
        boost::asio::ip::tcp::acceptor tmp_acceptor(tmp_io_context);

        boost::system::error_code tmp_error_code;

        tmp_acceptor.open(boost::asio::ip::tcp::v4(), tmp_error_code);

        if(tmp_error_code) { return(false); }

        tmp_acceptor.bind({boost::asio::ip::tcp::v4(), port_received}, tmp_error_code);

        return(tmp_error_code == boost::asio::error::address_in_use);
    }
    
    int Client::Done(void) const
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

    int Client::Get__Port(void) const { return(this->_port); }

    unsigned short Client::_Find_Opened_Ports(void)
    {
        for(unsigned short tmp_port_index(9061); tmp_port_index < 9079; ++tmp_port_index)
        {
            if(this->_Port_In_Use(tmp_port_index))
            {
                return(tmp_port_index);
            }
        }

        return(0);
    }
    
    std::string Client::Receive(void) const
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

    class Client *g_Client = nullptr;

    std::string g_outputs = "";

    DLL_EXTERNAL bool DLL_API API__Neural_Network__Is_Loaded(void) { return(true); }
    
    DLL_EXTERNAL bool DLL_API API__Neural_Network__Allocate(void)
    {
        // Client object need to be null.
        if(g_Client != nullptr) { return(false); }

        // Allocate client object.
        if((g_Client = new class Client) == nullptr) { return(false); }
            
        // Return success.
        return(true);
    }

    DLL_EXTERNAL bool DLL_API API__Neural_Network__Initialize(void)
    {
        // Client object need to be disconnected.
        if(g_Client->Connected()) { return(false); }

        // Connect to AI.
        if(g_Client->Connect() == false)
        {
            SAFE_DELETE(g_Client);

            return(false);
        }
            
        // Return success.
        return(true);
    }

    DLL_EXTERNAL bool DLL_API API__Neural_Network__Deinitialize(void)
    {
        // Client object need to not be null.
        if(g_Client == nullptr) { return(false); }

        // Deallocate client object.
        SAFE_DELETE(g_Client)

        // Return success.
        return(true);
    }
        
    DLL_EXTERNAL float DLL_API API__Neural_Network__Get__Loss(unsigned int const type_neural_network_use_received, unsigned int const type_loss_received)
    {
        // Client object need to not be null.
        if(g_Client == nullptr) { return(false); }

        // Query model history [2, 0, 1]
        g_Client->Send(QUERY_CODE::QUERY_MODEL_HISTORY + " <" + std::to_string(type_loss_received - 1) + ", 0, " + std::to_string(type_neural_network_use_received - 1) + "/>");
        if(g_Client->Done() == false) { return(false); }
            
        // Receive results.
        std::string const tmp_buffer(g_Client->Receive());
        if(g_Client->Done() == false) { return(false); }
        
        // Return results.
        return(std::stof(tmp_buffer));
    }
    
    DLL_EXTERNAL T_ DLL_API API__Neural_Network__Forward_Pass(T_ *const ptr_array_inputs_received)
    {
        // Client object need to not be null.
        if(g_Client == nullptr) { return(false); }

        // Buffer.
        int i(0);
            
        std::string tmp_buffer("");
            
        //  [0,     1,     2,     3,     4  ]
        //  [H_tm1, O_tm1, L_tm1, C_tm1, O_t]
        for(; i != 5; ++i) { tmp_buffer += std::to_string(ptr_array_inputs_received[i]) + " "; }

        tmp_buffer += std::to_string(ptr_array_inputs_received[i]);
        // |END| Buffer. |END|

        // Query predict Array.
        g_Client->Send(QUERY_CODE::QUERY_PREDICT + " <" + tmp_buffer + "/>");
        if(g_Client->Done() == false) { return(false); }
            
        // Receive results.
        tmp_buffer = g_Client->Receive();
        if(g_Client->Done() == false) { return(false); }
            
        // Return results.
        return(std::stof(tmp_buffer));
    }
}