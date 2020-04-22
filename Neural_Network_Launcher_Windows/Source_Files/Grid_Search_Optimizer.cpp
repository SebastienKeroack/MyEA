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
    #include <Enums/Enum_Type_Chart.hpp>

    #include <windows.h>
    #include <Form.hpp>
#endif

#include <Capturing/Shutdown/Shutdown.hpp>

#include <Grid_Search_Optimizer.hpp>

#include <Neural_Network/Grid_Search.hpp>

#include <iostream>

bool Grid_Search_Optimizer(class MyEA::Capturing::Shutdown &shutdown_module)
{
    std::string tmp_neural_network_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Neural network name: ";

    getline(std::cin, tmp_neural_network_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_neural_network_name + " - Grid Search").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager;
    
    tmp_Neural_Network_Manager.Set__Optimization_Auto_Save_Trainer(false);
    
    tmp_Neural_Network_Manager.Set__Optimization_Auto_Save_Competitor(false);
    
    tmp_Neural_Network_Manager.Set__Optimization_Auto_Save_Trained(false);

    if(tmp_Neural_Network_Manager.Set__Desired_Loss(0.0f) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Loss()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Initialize_Path(tmp_neural_network_name, tmp_neural_network_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_neural_network_name.c_str(),
                                 tmp_neural_network_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    if(tmp_Neural_Network_Manager.Initialize_Dataset_Manager() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    tmp_Neural_Network_Manager.Get__Dataset_Manager()->Set__Evaluation(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Evaluation on validation: ") ? MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION : MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    
    tmp_Neural_Network_Manager.Get__Dataset_Manager()->Set__Desired_Optimization_Time_Between_Reports(MyEA::String::Cin_Real_Number<double>(0.0, MyEA::Time::Date_Time_Now() + ": Desired optimization time between reports (seconds): "));
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    
    // Memory allocate.
    size_t const tmp_remaining_available_system_memory(Get__Remaining_Available_System_Memory(10.0L, KILOBYTE * KILOBYTE * KILOBYTE));

    PRINT_FORMAT("%s: Maximum available memory allocatable:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu] MBs." NEW_LINE, MyEA::Time::Date_Time_Now().c_str(), tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE);

    size_t const tmp_maximum_host_memory_allocate_bytes(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                               tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE,
                                                                                                                                               MyEA::Time::Date_Time_Now() + ": Maximum memory allocation (MBs): ") * KILOBYTE * KILOBYTE);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    // |END| Memory allocate. |END|

    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to load the neural network from a file?"))
    {
    #if defined(COMPILE_CUDA)
        size_t tmp_maximum_device_memory_allocate_bytes(0_zu);

        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use CUDA?"))
        {
            int tmp_index_device(-1);

            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            tmp_Neural_Network_Manager.Set__Use__CUDA(CUDA__Input__Use__CUDA(tmp_index_device, tmp_maximum_device_memory_allocate_bytes));
        }
    #endif
        
    #if defined(COMPILE_CUDA)
        if(tmp_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
                                                                                        tmp_maximum_host_memory_allocate_bytes,
                                                                                        tmp_maximum_device_memory_allocate_bytes,
                                                                                        false) == false)
    #else
        if(tmp_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
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
                                     tmp_maximum_device_memory_allocate_bytes,
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
    #if defined(COMPILE_CUDA)
        else if(tmp_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                               tmp_maximum_host_memory_allocate_bytes,
                                                                                               tmp_maximum_device_memory_allocate_bytes,
                                                                                               true) == false)
    #else
        else if(tmp_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                               tmp_maximum_host_memory_allocate_bytes,
                                                                                               true) == false)
    #endif
        {
        #if defined(COMPILE_CUDA)
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     tmp_maximum_device_memory_allocate_bytes,
                                     __LINE__);
        #else
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, true)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     __LINE__);
        #endif

            return(false);
        }
    }
    else
    {
        if(tmp_Neural_Network_Manager.Create_Neural_Network(tmp_maximum_host_memory_allocate_bytes) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Create_Neural_Network()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    class Grid_Search<T_> tmp_Grid_Search;
    
    if(tmp_Grid_Search.Input(tmp_Neural_Network_Manager) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Input(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(tmp_Grid_Search.Update_Tree() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update_Tree()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: FORM: Allocate." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    MyEA::Form::API__Form__Neural_Network__Allocate();

    PRINT_FORMAT("%s: FORM: Initialize chart of type grid search with 1 series." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH, 1u);

    tmp_Grid_Search.DataGridView_Initialize_Columns();
#endif
    
    if(shutdown_module.Create_Shutdown_Block(true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Create_Shutdown_Block(true)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Assign_Shutdown_Block(shutdown_module) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign_Shutdown_Block(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(shutdown_module.Get__On_Shutdown() == false
          &&
          shutdown_module.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Assign_Shutdown_Block(shutdown_module) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign_Shutdown_Block(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(shutdown_module.Get__On_Shutdown() == false
          &&
          shutdown_module.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(tmp_Grid_Search.Assign_Shutdown_Block(shutdown_module) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign_Shutdown_Block(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(shutdown_module.Get__On_Shutdown() == false
          &&
          shutdown_module.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(shutdown_module.Peak_Message_Async() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Peak_Message_Async()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(shutdown_module.Get__On_Shutdown() == false
          &&
          shutdown_module.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }

    if(tmp_Grid_Search.Search(tmp_Neural_Network_Manager) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Search(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(shutdown_module.Get__On_Shutdown() == false
          &&
          shutdown_module.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    
#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: FORM: Deallocate." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    MyEA::Form::API__Form__Neural_Network__Deallocate();
#endif

    PRINT_FORMAT("%s: Best hyper-parameters:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: %s" NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_COMPETITOR)->Get__Parameters().c_str());
    
    if(shutdown_module.Get__On_Shutdown() == false
      &&
      shutdown_module.Remove_Shutdown_Block() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}