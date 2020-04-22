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