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

#include <Configuration/Configuration.hpp>

#include <string>

#include <Reallocate/Reallocate.hpp>

#if defined(COMPILE_CUDA)
    #include <Tools/CUDA_Configuration.cuh>
#endif

#include <Neural_Network/Activation_Functions.hpp>
#include <Neural_Network/Data.hpp>
#include <Neural_Network/Dataset_Manager.hpp>
#include <Neural_Network/Internal.hpp>
#include <Neural_Network/Train.hpp>
