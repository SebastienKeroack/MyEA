#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>

#include <DLL/stdafx.hpp>

//#include "mysql_connection.h"

// CommonLib
#include <Tools/Message_Box.hpp>
#include <Tools/Time.hpp>
#include <Strings/String.hpp>

// NetworkLib
#include <CNetworkLib.hpp>

namespace MyEA
{
    namespace Network
    {
        bool Network_Connect_HTTP(bool const use_ssl_received,
                                                   unsigned int const try_count_received,
                                                   unsigned int const try_wait_milliseconds_received,
                                                   wchar_t const *const wc_ptr_url_received)
        {
            bool tmp_boolean(true);

            std::wstring const tmp_wstring(wc_ptr_url_received);
            std::string const tmp_url(tmp_wstring.begin(), tmp_wstring.end());

            WSADATA tmp_WSAData;
            SOCKET tmp_socket;
            struct sockaddr_in tmp_sockeTaddr_in;

            try
            {
                if(WSAStartup(MAKEWORD(2, 2), &tmp_WSAData) != 0) { throw("ERROR: From \"WSAStartup\", " + std::to_string(WSAGetLastError())); }

                inet_pton(AF_INET, tmp_url.c_str(), &tmp_sockeTaddr_in.sin_addr.s_addr);

                unsigned short const tmp_port = use_ssl_received ? static_cast<unsigned short>(443) : static_cast<unsigned short>(80);

                tmp_sockeTaddr_in.sin_family = AF_INET; // AF_INET = IPv4, SOCK_STREAM = TCP protocol
                tmp_sockeTaddr_in.sin_port = htons(tmp_port);

                if((tmp_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) { throw("ERROR: Could not create socket: " + std::to_string(WSAGetLastError())); }
                
                int tmp_return_connect;

                for(unsigned int i(0u); i != try_count_received; ++i)
                {
                    tmp_return_connect = 0;

                    if((tmp_return_connect = connect(tmp_socket, (struct sockaddr *)&tmp_sockeTaddr_in, sizeof(tmp_sockeTaddr_in))) != INVALID_SOCKET) { break; }
                    else if(i == (try_count_received - 1u)) { throw("ERROR: Not connected to: " + tmp_url + ":" + std::to_string(tmp_port) + ", "+ std::to_string(tmp_return_connect)); }
                    else { MyEA::Time::Sleep(try_wait_milliseconds_received); }
                }
            }
            catch(std::string const &ref_exception_received)
            {
                tmp_boolean = false;

                MyEA::Common::Message_Box__OK("Error : " + ref_exception_received, "Network");
            }

            closesocket(tmp_socket);

            WSACleanup();

            return(tmp_boolean);
        }

        bool Network_Connect(unsigned short const port_received,
                                          unsigned int const try_count_received,
                                          unsigned int const try_wait_milliseconds_received,
                                          wchar_t const* const wc_ptr_url_received)
        {
            bool tmp_boolean(true);

            std::wstring const tmp_wstring(wc_ptr_url_received);
            std::string const tmp_url(tmp_wstring.begin(), tmp_wstring.end());

            WSADATA tmp_WSAData;
            SOCKET tmp_socket;
            struct sockaddr_in tmp_sockeTaddr_in;

            try
            {
                if(WSAStartup(MAKEWORD(2, 2), &tmp_WSAData) != 0) { throw("ERROR: From \"WSAStartup\", " + std::to_string(WSAGetLastError())); }

                inet_pton(AF_INET, tmp_url.c_str(), &tmp_sockeTaddr_in.sin_addr.s_addr);

                tmp_sockeTaddr_in.sin_family = AF_INET; // AF_INET = IPv4, SOCK_STREAM = TCP protocol
                tmp_sockeTaddr_in.sin_port = htons(port_received);
                
                if((tmp_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) { throw("ERROR: Could not create socket: " + std::to_string(WSAGetLastError())); }

                //bind(tmp_socket, (SOCKADDR *)&tmp_sockeTaddr_in, sizeof(tmp_sockeTaddr_in));

                //listen(tmp_socket, 0);
                
                int tmp_return_connect;

                for(unsigned int i(0u); i != try_count_received; ++i)
                {
                    tmp_return_connect = 0;

                    if((tmp_return_connect = connect(tmp_socket, (struct sockaddr *)&tmp_sockeTaddr_in, sizeof(tmp_sockeTaddr_in))) != INVALID_SOCKET) { break; }
                    else if(i == (try_count_received - 1u)) { throw("ERROR: Not connected to: " + tmp_url + ":" + std::to_string(port_received) + ", "+ std::to_string(tmp_return_connect)); }
                    else { MyEA::Time::Sleep(try_wait_milliseconds_received); }
                }
            }
            catch(std::string const &ref_exception_received)
            {
                tmp_boolean = false;

                MyEA::Common::Message_Box__OK("Error : " + ref_exception_received, "Network");
            }

            closesocket(tmp_socket);

            WSACleanup();

            return(tmp_boolean);
        }
    }
}