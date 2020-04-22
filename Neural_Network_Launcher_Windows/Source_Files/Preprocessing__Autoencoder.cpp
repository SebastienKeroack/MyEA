#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Preprocessing__Autoencoder.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Autoencoder(void)
{
    std::string tmp_autoencoder_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Autoencoder name: ";

    getline(std::cin, tmp_autoencoder_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_autoencoder_name + " - Preprocessing, Autoencoder").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager;
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_autoencoder_name, tmp_autoencoder_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_autoencoder_name.c_str(),
                                 tmp_autoencoder_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_storage = 0;
    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_Neural_Network_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    // Memory allocate.
    size_t const tmp_remaining_available_system_memory(Get__Remaining_Available_System_Memory(10.0L, KILOBYTE * KILOBYTE * KILOBYTE));

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Maximum available memory allocatable." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu] MBs." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE);

    size_t const tmp_maximum_host_memory_allocate_bytes(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                               tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE,
                                                                                                                                               MyEA::Time::Date_Time_Now() + ": Maximum memory allocation (MBs): ") * 1024u * 1024u);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    // |END| Memory allocate. |END|

    class Neural_Network *tmp_ptr_Neural_Network(nullptr);

    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to load the neural network (Tainer) from a file?"))
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

        tmp_ptr_Neural_Network = tmp_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);
    }
    else
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

        tmp_ptr_Neural_Network = tmp_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED);
    }
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_autoencoder_name, tmp_autoencoder_name + "_AE") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_AE)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_autoencoder_name.c_str(),
                                 tmp_autoencoder_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Save(tmp_ptr_Neural_Network, tmp_Neural_Network_Manager.Get__Path_Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str(),
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    return(true);
}
