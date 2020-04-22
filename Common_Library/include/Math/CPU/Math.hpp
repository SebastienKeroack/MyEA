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

namespace MyEA::Math
{
    size_t Recursive_Fused_Multiply_Add(size_t const *const ptr_array_value_received,
                                        size_t const depth_received,
                                        size_t const depth_end_received);

    template<typename T>
    T Reverse_Integer(T const integer_received);

    template<typename T>
    bool Is_NaN(T const value_received);

    template<typename T>
    T Sign(T const value_received)
    {
        return(static_cast<T>(T(0) < value_received) - static_cast<T>(value_received < T(0)));
    }

    template<typename T>
    T Absolute(T const value_received)
    {
        return(value_received >= 0 ? value_received : -value_received);
    }

    template<typename T>
    T Maximum(T const x_received, T const y_received)
    {
        return(x_received > y_received ? x_received : y_received);
    }

    template<typename T>
    T Minimum(T const x_received, T const y_received)
    {
        return(x_received < y_received ? x_received : y_received);
    }

    template<typename T>
    T Clip(T const value_received,
           T const minimum_received,
           T const maximum_received)
    {
        return((value_received < minimum_received) ? minimum_received
                                                     :
                                                     ((value_received > maximum_received) ? maximum_received
                                                                                            :
                                                                                            value_received));
    }
}