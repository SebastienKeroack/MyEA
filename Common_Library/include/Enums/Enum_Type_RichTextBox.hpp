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

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_RICHTEXTBOX
        {
            TYPE_RTB_NONE = 0,
            TYPE_RTB_OUTPUT = 1,
            TYPE_RTB_STATUS_GENERAL = 2,
            TYPE_RTB_STATUS_PRICE = 3,
            TYPE_RTB_STATUS_MONEY = 4,
            TYPE_RTB_STATUS_NETWORK = 5
        };
    }
}