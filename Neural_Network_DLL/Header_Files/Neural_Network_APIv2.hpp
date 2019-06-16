#include <Tools/Configuration.hpp>

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

    DLL_EXTERNAL bool DLL_API API__Neural_Network__Is_Loaded(void);
        
    DLL_EXTERNAL bool DLL_API API__Neural_Network__Allocate(void);

    DLL_EXTERNAL bool DLL_API API__Neural_Network__Initialize(bool const is_type_position_long_received,
                                                              unsigned int const type_indicator_received,
                                                              unsigned int const time_frames_received);
        
    DLL_EXTERNAL bool DLL_API API__Neural_Network__Deinitialize(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
    DLL_EXTERNAL float DLL_API API__Neural_Network__Get__Loss(bool const is_type_position_long_received,
                                                              unsigned int const type_indicator_received,
                                                              unsigned int const type_neural_network_use_received,
                                                              unsigned int const type_loss_received);
        
    DLL_EXTERNAL float DLL_API API__Neural_Network__Get__Accuracy(bool const is_type_position_long_received,
                                                                  unsigned int const type_indicator_received,
                                                                  unsigned int const type_neural_network_use_received,
                                                                  unsigned int const type_accuracy_received);
        
    DLL_EXTERNAL T_ DLL_API API__Neural_Network__Forward_Pass(bool const is_type_position_long_received,
                                                              unsigned int const type_indicator_received,
                                                              unsigned int const type_neural_network_use_received,
                                                              T_ *const ptr_array_inputs_received);
}