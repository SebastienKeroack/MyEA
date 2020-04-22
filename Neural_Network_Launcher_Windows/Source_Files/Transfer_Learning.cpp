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
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Transfer_Learning.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Transfer_Learning(void)
{
    std::string tmp_source_neural_network_name,
                   tmp_destination_neural_network_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Neural network, source name: ";

    getline(std::cin, tmp_source_neural_network_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Neural network, destination name: ";

    getline(std::cin, tmp_destination_neural_network_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_source_neural_network_name + " Transfer learning - Neural Network").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_source_Neural_Network_Manager,
                                                                                      tmp_destination_Neural_Network_Manager;
    
    if(tmp_source_Neural_Network_Manager.Initialize_Path(tmp_source_neural_network_name, tmp_source_neural_network_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_source_neural_network_name.c_str(),
                                 tmp_source_neural_network_name.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_destination_Neural_Network_Manager.Initialize_Path(tmp_destination_neural_network_name, tmp_destination_neural_network_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_destination_neural_network_name.c_str(),
                                 tmp_destination_neural_network_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Memory allocate.
    size_t const tmp_remaining_available_system_memory(Get__Remaining_Available_System_Memory(10.0L, KILOBYTE * KILOBYTE * KILOBYTE));

    PRINT_FORMAT("%s: Maximum available memory allocatable:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu] MBs." NEW_LINE, MyEA::Time::Date_Time_Now().c_str(), tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE);

    size_t const tmp_maximum_host_memory_allocate_bytes(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                               tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE,
                                                                                                                                               MyEA::Time::Date_Time_Now() + ": Maximum memory allocation (MBs): ") * 1024u * 1024u);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    // |END| Memory allocate. |END|

    class Neural_Network *tmp_ptr_source_Neural_Network,
                                    *tmp_ptr_destination_Neural_Network;

    // Load source neural network.
#if defined(COMPILE_CUDA)
    if(tmp_source_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                                tmp_maximum_host_memory_allocate_bytes,
                                                                                                0_zu,
                                                                                                false) == false)
#else
    if(tmp_source_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                                tmp_maximum_host_memory_allocate_bytes,
                                                                                                false) == false)
#endif
    {
    #if defined(COMPILE_CUDA)
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED].c_str(),
                                 tmp_maximum_host_memory_allocate_bytes,
                                 0_zu,
                                 __LINE__);
    #else
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED].c_str(),
                                 tmp_maximum_host_memory_allocate_bytes,
                                 __LINE__);
    #endif

        return(false);
    }
    
    tmp_ptr_source_Neural_Network = tmp_source_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED);
    // |END| Load source neural network. |END|
    
    // Create/Load destination neural network.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to load the neural network (destination) from a file?"))
    {
    #if defined(COMPILE_CUDA)
        if(tmp_destination_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
                                                                                                          tmp_maximum_host_memory_allocate_bytes,
                                                                                                          0_zu,
                                                                                                          false) == false)
    #else
        if(tmp_destination_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
                                                                                                          tmp_maximum_host_memory_allocate_bytes,
                                                                                                          false) == false)
    #endif
        {
        #if defined(COMPILE_CUDA)
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     0_zu,
                                     __LINE__);
        #else
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     __LINE__);
        #endif

            return(false);
        }
    }
    else if(tmp_destination_Neural_Network_Manager.Create_Neural_Network(tmp_maximum_host_memory_allocate_bytes) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Create_Neural_Network(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_maximum_host_memory_allocate_bytes,
                                 __LINE__);

        return(false);
    }
    
    tmp_ptr_destination_Neural_Network = tmp_destination_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);
    // |END| Create/Load destination neural network. |END|
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Transfer learning... ", MyEA::Time::Date_Time_Now().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_ptr_source_Neural_Network->Transfer_Learning(tmp_ptr_destination_Neural_Network) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Transfer_Learning(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s.(net, nn)... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_destination_Neural_Network_Manager.Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "").c_str());
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_destination_Neural_Network_Manager.Save_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_Neural_Network(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL].c_str(),
                                 __LINE__);
        
        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE);

    return(true);
}
