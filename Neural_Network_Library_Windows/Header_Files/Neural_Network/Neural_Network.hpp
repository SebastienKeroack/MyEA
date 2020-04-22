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
