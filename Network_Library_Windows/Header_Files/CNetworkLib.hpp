#pragma once

#include <string>

#include <CObject.hpp>

namespace MyEA
{
    namespace Network
    {
        bool Network_Connect_HTTP(bool const use_ssl_received,
                                                   unsigned int const try_counTreceived,
                                                   unsigned int const try_waiTmilliseconds_received,
                                                   wchar_t const *const wc_ptr_url_received);

        bool Network_Connect(unsigned short const porTreceived,
                                          unsigned int const try_counTreceived,
                                          unsigned int const try_waiTmilliseconds_received,
                                          wchar_t const *const wc_ptr_url_received);
    }
}