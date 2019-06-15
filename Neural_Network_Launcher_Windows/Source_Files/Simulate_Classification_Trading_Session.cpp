#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Simulate_Classification_Trading_Session.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Simulate_Classification_Trading_Session(void)
{
    std::string tmp_neural_network_name;
    
    std::cout << MyEA::String::Get__Time() << ": Neural network name: ";

    getline(std::cin, tmp_neural_network_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_neural_network_name + " - Simulate Classification Trading Session").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE);
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_neural_network_name, tmp_neural_network_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_neural_network_name.c_str(),
                                 tmp_neural_network_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_Neural_Network_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Dataset Manager Parameters. |END|

    // Memory allocate.
    size_t const tmp_remaining_available_system_memory(Get__Remaining_Available_System_Memory(10.0L, KILOBYTE * KILOBYTE * KILOBYTE));

    PRINT_FORMAT("%s: Maximum available memory allocatable:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu] MBs." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE);

    size_t const tmp_maximum_host_memory_allocate_bytes(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                               tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE,
                                                                                                                                               MyEA::String::Get__Time() + ": Maximum memory allocation (MBs): ") * 1024u * 1024u);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    // |END| Memory allocate. |END|

    class Neural_Network *tmp_ptr_Neural_Network(nullptr);

    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to load the neural network (Tainer) from a file?"))
    {
    #if defined(COMPILE_CUDA)
        size_t tmp_maximum_device_memory_allocate_bytes(0_zu);

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use CUDA?"))
        {
            int tmp_index_device(-1);

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Set__Use__CUDA(CUDA__Input__Use__CUDA(tmp_index_device, tmp_maximum_device_memory_allocate_bytes));
        }

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
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     tmp_maximum_device_memory_allocate_bytes,
                                     __LINE__);
        #else
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     __LINE__);
        #endif

            return(false);
        }

        tmp_ptr_Neural_Network = tmp_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);
    }
    else
    {
    #if defined(COMPILE_CUDA)
        size_t tmp_maximum_device_memory_allocate_bytes(0_zu);

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use CUDA?"))
        {
            int tmp_index_device(-1);

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Set__Use__CUDA(CUDA__Input__Use__CUDA(tmp_index_device, tmp_maximum_device_memory_allocate_bytes));
        }

        if(tmp_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                        tmp_maximum_host_memory_allocate_bytes,
                                                                                        tmp_maximum_device_memory_allocate_bytes,
                                                                                        true) == false)
    #else
        if(tmp_Neural_Network_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                        tmp_maximum_host_memory_allocate_bytes,
                                                                                        true) == false)
    #endif
        {
        #if defined(COMPILE_CUDA)
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     tmp_maximum_device_memory_allocate_bytes,
                                     __LINE__);
        #else
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Neural_Network(%s, %zu, true)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED].c_str(),
                                     tmp_maximum_host_memory_allocate_bytes,
                                     __LINE__);
        #endif
            
            return(false);
        }

        tmp_ptr_Neural_Network = tmp_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED);
    }
    
    switch(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Type_Storage())
    {
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
            PRINT_FORMAT("%s: Training dataset:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Simulate_Classification_Trading_Session(tmp_ptr_Neural_Network);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
            PRINT_FORMAT("%s: Training dataset:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Simulate_Classification_Trading_Session(tmp_ptr_Neural_Network);

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Testing dataset:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Simulate_Classification_Trading_Session(tmp_ptr_Neural_Network);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
            PRINT_FORMAT("%s: Training dataset:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Simulate_Classification_Trading_Session(tmp_ptr_Neural_Network);
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Validating dataset:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Simulate_Classification_Trading_Session(tmp_ptr_Neural_Network);
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Testing dataset:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Simulate_Classification_Trading_Session(tmp_ptr_Neural_Network);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Type_Storage(),
                                     MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Type_Storage()].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}
