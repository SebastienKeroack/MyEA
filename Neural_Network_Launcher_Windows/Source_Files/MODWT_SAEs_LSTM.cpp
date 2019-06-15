#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <windows.h>
    #include <Form.hpp>
#endif

#if defined(COMPILE_UI)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <Form.hpp>
#endif

#include <MODWT_SAEs_LSTM.hpp>

#include <Tools/Shutdown_Block.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

//#define USE_MODWT

bool Preprocessing__SAEs(class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
{
    size_t tmp_index;

    T_ tmp_minimum_input,
         tmp_maximum_input;
            
    class Dataset<T_> *const tmp_ptr_TrainingSet(ptr_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));

    // Price.
    tmp_minimum_input = (std::numeric_limits<ST_>::max)();
    tmp_maximum_input = -(std::numeric_limits<ST_>::max)();

    // Price, MODWT.
#if defined(USE_MODWT)
    for(tmp_index = 0_zu; tmp_index != 4_zu; ++tmp_index)
    {
        if(ptr_Dataset_Manager_received->Preprocessing__MODWT(tmp_index,
                                                                                                3_zu,
                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        if(ptr_Dataset_Manager_received->Preprocessing__MODWT(tmp_index,
                                                                                                3_zu,
                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
#endif
    
    // Price, Get Min-Max.
    for(tmp_index = 0_zu; tmp_index != 8_zu; ++tmp_index)
    {
        tmp_minimum_input = MyEA::Math::Minimum<T_>(tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                                                   tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                                                   tmp_index,
                                                                                                                                                   ENUM_TYPE_INPUT::TYPE_INPUT_INPUT), tmp_minimum_input);
                        
        tmp_maximum_input = MyEA::Math::Maximum<T_>(tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                                                      tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                                                      tmp_index,
                                                                                                                                                      ENUM_TYPE_INPUT::TYPE_INPUT_INPUT), tmp_maximum_input);
    }

    // Price, Set Min-Max.
    for(tmp_index = 0_zu; tmp_index != 8_zu; ++tmp_index)
    {
        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                 ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                 tmp_index,
                                                                                                                 tmp_minimum_input,
                                                                                                                 tmp_maximum_input,
                                                                                                                 0_T,
                                                                                                                 1_T,
                                                                                                                 ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                 ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                 tmp_index,
                                                                                                                 tmp_minimum_input,
                                                                                                                 tmp_maximum_input,
                                                                                                                 0_T,
                                                                                                                 1_T,
                                                                                                                 ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    // |END| Price. |END|

    // RSI.
    if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                             ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                             9_zu,
                                                                                                             0_T,
                                                                                                             100_T,
                                                                                                             0_T,
                                                                                                             1_T,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                             ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                             9_zu,
                                                                                                             0_T,
                                                                                                             100_T,
                                                                                                             0_T,
                                                                                                             1_T,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| RSI. |END|
    
    // ATR.
    tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                     tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                     10_zu,
                                                                                                     ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
    
    tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                       tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                       10_zu,
                                                                                                       ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

    if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                             ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                             10_zu,
                                                                                                             tmp_minimum_input,
                                                                                                             tmp_maximum_input,
                                                                                                             0_T,
                                                                                                             1_T,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                             ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                             10_zu,
                                                                                                             tmp_minimum_input,
                                                                                                             tmp_maximum_input,
                                                                                                             0_T,
                                                                                                             1_T,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| ATR. |END|
    
    // StdDev.
    tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                     tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                     11_zu,
                                                                                                     ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
    
    tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                       tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                       11_zu,
                                                                                                       ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

    if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                             ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                             11_zu,
                                                                                                             tmp_minimum_input,
                                                                                                             tmp_maximum_input,
                                                                                                             0_T,
                                                                                                             1_T,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                             ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                             11_zu,
                                                                                                             tmp_minimum_input,
                                                                                                             tmp_maximum_input,
                                                                                                             0_T,
                                                                                                             1_T,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| StdDev. |END|

    return(true);
}

bool MODWT_SAEs_LSTM(class Shutdown_Block &ref_Shutdown_Block_received)
{
    std::string tmp_finance_name,
                   tmp_sae_name,
                   tmp_lstm_name;
    
    std::cout << MyEA::String::Get__Time() << ": Financial dataset name: ";

    getline(std::cin, tmp_finance_name);

    std::cout << MyEA::String::Get__Time() << ": SAEs name: ";

    getline(std::cin, tmp_sae_name);

    std::cout << MyEA::String::Get__Time() << ": LSTM name: ";

    getline(std::cin, tmp_lstm_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_finance_name + " - " + tmp_sae_name + " - " + tmp_lstm_name + " | MODWT-SAEs-LSTM").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Financial_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE),
                                                                                       tmp_SAE_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE),
                                                                                       tmp_LSTM_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE);
    
    if(tmp_Financial_Manager.Initialize_Path(tmp_finance_name, tmp_finance_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_finance_name.c_str(),
                                 tmp_finance_name.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_SAE_Manager.Initialize_Path(tmp_sae_name, tmp_sae_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_sae_name.c_str(),
                                 tmp_sae_name.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_LSTM_Manager.Initialize_Path(tmp_lstm_name, tmp_lstm_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_lstm_name.c_str(),
                                 tmp_lstm_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;
    
    tmp_Dataset_Manager_Parameters.type_storage = 0;
    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_Financial_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    tmp_Dataset_Manager_Parameters.type_storage = 2;
    tmp_Dataset_Manager_Parameters.type_training = 1;

    tmp_Dataset_Manager_Parameters.percent_training_size = 94.0;
    tmp_Dataset_Manager_Parameters.percent_validation_size = 4.0;

    tmp_Dataset_Manager_Parameters.training_parameters.value_0 = true;
    tmp_Dataset_Manager_Parameters.training_parameters.value_1 = 116;
    tmp_Dataset_Manager_Parameters.training_parameters.value_2 = 0;
    
    if(tmp_SAE_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_LSTM_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
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

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Maximum available memory allocatable:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu] MBs." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE);

    size_t const tmp_maximum_host_memory_allocate_bytes(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                               tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE,
                                                                                                                                               MyEA::String::Get__Time() + ": Maximum memory allocation (MBs): ") * 1024u * 1024u);
    // |END| Memory allocate. |END|
    
    // CUDA.
#if defined(COMPILE_CUDA)
    size_t tmp_maximum_device_memory_allocate_bytes(0_zu);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use CUDA?"))
    {
        int tmp_index_device(-1);

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        tmp_SAE_Manager.Set__Use__CUDA(CUDA__Input__Use__CUDA(tmp_index_device, tmp_maximum_device_memory_allocate_bytes));
    }
#endif
    // |END| CUDA. |END|
    
    class Neural_Network *tmp_ptr_SAE(nullptr),
                                     *tmp_ptr_LSTM(nullptr);

    enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const tmp_type_neural_network_use(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

    bool const tmp_append_to_dataset(false);

    // Load SAEs.
    {
    #if defined(COMPILE_CUDA)
        if(tmp_SAE_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
                                                                                        tmp_maximum_host_memory_allocate_bytes,
                                                                                        tmp_maximum_device_memory_allocate_bytes,
                                                                                        false) == false)
    #else
        if(tmp_SAE_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
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
    #if defined(COMPILE_CUDA)
        else if(tmp_SAE_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                               tmp_maximum_host_memory_allocate_bytes,
                                                                               tmp_maximum_device_memory_allocate_bytes,
                                                                               true) == false)
    #else
        else if(tmp_SAE_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
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

        tmp_ptr_SAE = tmp_SAE_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED);
    }
    // |END| Load SAEs. |END|
    
    // Setup SAEs input/output mode.
    if(tmp_ptr_SAE->Set__Input_Mode(true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(true)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_SAE->Set__Output_Mode(false) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    // Load LSTM.
    {
    #if defined(COMPILE_CUDA)
        if(tmp_LSTM_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
                                                                                        tmp_maximum_host_memory_allocate_bytes,
                                                                                        tmp_maximum_device_memory_allocate_bytes,
                                                                                        false) == false)
    #else
        if(tmp_LSTM_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER,
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
    #if defined(COMPILE_CUDA)
        else if(tmp_LSTM_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
                                                                                 tmp_maximum_host_memory_allocate_bytes,
                                                                                 tmp_maximum_device_memory_allocate_bytes,
                                                                                 true) == false)
    #else
        else if(tmp_LSTM_Manager.Load_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED,
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

        tmp_ptr_LSTM = tmp_LSTM_Manager.Get__Neural_Network(tmp_type_neural_network_use);
    }
    // |END| Load LSTM. |END|
    
    class Dataset_Manager<T_> const *const tmp_ptr_Financial_Dataset(tmp_Financial_Manager.Get__Dataset_Manager());
    class Dataset_Manager<T_> *const tmp_ptr_SAEs_Dataset(tmp_SAE_Manager.Get__Dataset_Manager()),
                                               *const tmp_ptr_LSTM_Dataset(tmp_LSTM_Manager.Get__Dataset_Manager());
    
    // Validate input(s)/output(s) size.
    if(tmp_ptr_LSTM_Dataset->Get__Number_Examples() == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_LSTM_Dataset->Get__Number_Examples() != tmp_ptr_SAEs_Dataset->Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of data (%zu) differ from the number of data received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_LSTM_Dataset->Get__Number_Examples(),
                                 tmp_ptr_SAEs_Dataset->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_LSTM_Dataset->Get__Number_Recurrent_Depth() != tmp_ptr_SAEs_Dataset->Get__Number_Recurrent_Depth())
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of recurrent depth (%zu) differ from the number of recurrent depth received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_LSTM_Dataset->Get__Number_Recurrent_Depth(),
                                 tmp_ptr_SAEs_Dataset->Get__Number_Recurrent_Depth(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_SAEs_Dataset->Check_Topology(tmp_ptr_SAE->number_inputs,
                                                                            tmp_ptr_SAE->number_outputs,
                                                                            tmp_ptr_SAE->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_SAE->number_inputs,
                                 tmp_ptr_SAE->number_outputs,
                                 tmp_ptr_SAE->number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_SAE->type_network != MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        PRINT_FORMAT("%s: %s: ERROR: The neural network (%s) receive as argument need to be a %s. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[tmp_ptr_SAE->type_network].c_str(),
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER].c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_LSTM_Dataset->Check_Topology(tmp_ptr_LSTM->number_inputs,
                                                                            tmp_ptr_LSTM->number_outputs,
                                                                            tmp_ptr_LSTM->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_LSTM->number_inputs,
                                 tmp_ptr_LSTM->number_outputs,
                                 tmp_ptr_LSTM->number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_LSTM_Dataset->Get__Number_Inputs() != tmp_ptr_SAE->Get__Output_Size())
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of input(s) (%zu) differ from the number of output(s) from the autoencoder (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_LSTM_Dataset->Get__Number_Inputs(),
                                 tmp_ptr_SAE->Get__Output_Size(),
                                 __LINE__);

        return(false);
    }
    // |END| Validate input(s)/output(s) size. |END|
    
    // Input for optimization.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    bool const tmp_optimization(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to optimize?"));

    size_t tmp_optimization_time_sae(1_zu),
              tmp_optimization_time_lstm(1_zu);
    
    struct MyEA::Common::While_Condition tmp_while_condition;

    if(tmp_optimization)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: SAEs optimization time." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=120 (seconds)." NEW_LINE, MyEA::String::Get__Time().c_str());
        tmp_optimization_time_sae = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Time in seconds: ");
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: LSTM optimization time." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=120 (seconds)." NEW_LINE, MyEA::String::Get__Time().c_str());
        tmp_optimization_time_lstm = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Time in seconds: ");

        tmp_while_condition.type_while_condition = MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_EXPIRATION;
    }
    // |END| Input for optimization. |END|

    // Preprocess SAEs dataset.
    if(Preprocessing__SAEs(tmp_ptr_SAEs_Dataset) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__SAEs(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    size_t const tmp_number_examples(tmp_ptr_Financial_Dataset->Get__Number_Examples()),
                       tmp_number_recurrent_depth(tmp_ptr_Financial_Dataset->Get__Number_Recurrent_Depth());
    size_t tmp_index,
              tmp_example_index,
              tmp_time_step,
              tmp_time_average,
              tmp_count_wins(0_zu),
              tmp_count_lossses(0_zu);

    T_ const *tmp_ptr_array_financial_inputs;
    T_ **tmp_ptr_matrix_inputs,
         **tmp_ptr_matrix_outputs,
         *tmp_ptr_array_sae_inputs,
         *tmp_ptr_array_lstm_inputs,
         *tmp_ptr_array_lstm_outputs,
         tmp_minimum_ohlc_input,
         tmp_maximum_ohlc_input,
         tmp_summation_loss(0),
         tmp_online_loss(0),
         tmp_online_accuracy(0),
         tmp_EMA;
    
    std::thread tmp_thread_SAE,
                    tmp_thread_LSTM;

    // EMA.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Exponential moving average." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1 , - ]." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=24." NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_time_average = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Time average: ");
    tmp_EMA = tmp_time_average == 0_zu ? 0_T : 1_T / static_cast<T_>(tmp_time_average);
    // |END| EMA. |END|

    if((tmp_ptr_matrix_inputs = new T_*[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_*),
                                 __LINE__);

        return(false);
    }
    
    if((tmp_ptr_matrix_outputs = new T_*[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_*),
                                 __LINE__);

        return(false);
    }
    
    if((tmp_ptr_array_sae_inputs = new T_[tmp_ptr_SAE->number_inputs * tmp_ptr_SAE->number_recurrent_depth]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_SAE->number_inputs * tmp_ptr_SAE->number_recurrent_depth * sizeof(T_),
                                 __LINE__);

        return(false);
    }
    
    if((tmp_ptr_array_lstm_inputs = new T_[tmp_ptr_LSTM->number_inputs * tmp_ptr_LSTM->number_recurrent_depth]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_LSTM->number_inputs * tmp_ptr_LSTM->number_recurrent_depth * sizeof(T_),
                                 __LINE__);

        return(false);
    }
    
    if((tmp_ptr_array_lstm_outputs = new T_[tmp_ptr_LSTM->number_outputs * tmp_ptr_LSTM->number_recurrent_depth]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_LSTM->number_outputs * tmp_ptr_LSTM->number_recurrent_depth * sizeof(T_),
                                 __LINE__);

        return(false);
    }
    
#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    bool const tmp_print_chart(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to print the chart?"));

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: FORM: Allocate." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Allocate();

    PRINT_FORMAT("%s: FORM: Initialize chart of type loss with 2 series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS, 2u);

    PRINT_FORMAT("%s: FORM: Initialize chart of type accuracy with 2 series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY, 2u);
    
    PRINT_FORMAT("%s: FORM: Initialize chart of type output with 2 series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT, 2u);

    PRINT_FORMAT("%s: FORM: Enable training series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Use_Datapoint_Training(true);

    PRINT_FORMAT("%s: FORM: Initialize chart means at 100 datapoint." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Total_Means(100u);

    if(tmp_print_chart)
    {
        PRINT_FORMAT("%s: FORM: Maximum ploted example(s)." NEW_LINE, MyEA::String::Get__Time().c_str());
        if(tmp_ptr_SAEs_Dataset->User_Controls__Set__Maximum_Ploted_Example() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Maximum_Ploted_Example()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        tmp_ptr_LSTM_Dataset->Set__Maximum_Ploted_Examples(tmp_ptr_SAEs_Dataset->Get__Maximum_Ploted_Examples());
        
        PRINT_FORMAT("%s: FORM: Time delay ploted." NEW_LINE, MyEA::String::Get__Time().c_str());
        if(tmp_ptr_SAEs_Dataset->User_Controls__Set__Time_Delay_Ploted() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Time_Delay_Ploted()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        else if(tmp_ptr_LSTM_Dataset->Set__Time_Delay_Ploted(tmp_ptr_SAEs_Dataset->Get__Time_Delay_Ploted()) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Time_Delay_Ploted(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_SAEs_Dataset->Get__Time_Delay_Ploted(),
                                     __LINE__);

            return(false);
        }

        PRINT_FORMAT("%s: FORM: Plot dataset manager." NEW_LINE, MyEA::String::Get__Time().c_str());
        if(tmp_ptr_SAEs_Dataset->Plot__Dataset_Manager() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        tmp_SAE_Manager.Get__Dataset_Manager()->Set__Plot__Output(false);

        tmp_LSTM_Manager.Get__Dataset_Manager()->Set__Plot__Output(false);
    }
#endif
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
    {
        // Get inputs (financial data).
        tmp_ptr_array_financial_inputs = tmp_ptr_Financial_Dataset->Get__Input_At(tmp_example_index);
        memcpy(tmp_ptr_array_sae_inputs,
                     tmp_ptr_array_financial_inputs,
                     tmp_number_recurrent_depth * tmp_ptr_Financial_Dataset->Get__Number_Inputs() * sizeof(T_));

        // Preprocess SAEs.
        {
            tmp_minimum_ohlc_input = tmp_ptr_SAEs_Dataset->Get__Scalar__Minimum_Maximum(ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)[0u].minimum_value;
            tmp_maximum_ohlc_input = tmp_ptr_SAEs_Dataset->Get__Scalar__Minimum_Maximum(ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)[0u].maximum_value;

            // Price.
            //  Price, Min-Max.
            for(tmp_index = 0_zu; tmp_index != 8_zu; ++tmp_index)
            {
                if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum(tmp_index,
                                                                                                              tmp_ptr_array_sae_inputs,
                                                                                                              ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            }
            
        #if defined(USE_MODWT)
            //  Price, MODWT.
            for(tmp_index = 0_zu; tmp_index != 4_zu; ++tmp_index)
            {
                // Inverse preprocess min-max before preprocessing modwt.
                if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(tmp_index, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                
                // Inverse preprocess modwt before preprocessing the inputs receive as argument.
                if(tmp_ptr_SAEs_Dataset->Preprocessing__MODWT_Inverse(tmp_index, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT_Inverse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                // Preprocess array receive as arguments to remove noise.
                if(tmp_ptr_SAEs_Dataset->Preprocessing__MODWT(tmp_index,
                                                                                              3_zu,
                                                                                              tmp_ptr_array_sae_inputs,
                                                                                              ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                
                // Re-preprocess dataset to remove noise.
                if(tmp_ptr_SAEs_Dataset->Preprocessing__MODWT(tmp_index,
                                                                                              3_zu,
                                                                                              ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                // Re-preprocess min-max with past parameters after preprocessing modwt.
                if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                              tmp_ptr_SAEs_Dataset->Get__Number_Examples(),
                                                                                                              tmp_index,
                                                                                                              tmp_minimum_ohlc_input,
                                                                                                              tmp_maximum_ohlc_input,
                                                                                                              0_T,
                                                                                                              1_T,
                                                                                                              ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            }
        #endif
            // |END| Price. |END|

            // RSI.
            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum(9_zu,
                                                                                                          tmp_ptr_array_sae_inputs,
                                                                                                          ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
            // |END| RSI. |END|

            // ATR.
            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum(10_zu,
                                                                                                          tmp_ptr_array_sae_inputs,
                                                                                                          ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
            // |END| ATR. |END|

            // StdDev.
            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum(11_zu,
                                                                                                          tmp_ptr_array_sae_inputs,
                                                                                                          ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
            // |END| StdDev. |END|
        }
        // |END| Preprocess SAEs. |END|
        
        // Propagate inputs into the SAEs.
        tmp_ptr_matrix_inputs[0u] = tmp_ptr_array_sae_inputs;
        tmp_ptr_SAE->Forward_Pass(1_zu, tmp_ptr_matrix_inputs);
        
        // Get inputs (SAEs outputs).
        for(tmp_time_step = 0_zu; tmp_time_step != tmp_number_recurrent_depth; ++tmp_time_step)
        {
            memcpy(tmp_ptr_array_lstm_inputs + tmp_time_step * tmp_ptr_LSTM->number_inputs,
                         tmp_ptr_SAE->Get__Outputs(0_zu, tmp_time_step),
                         tmp_ptr_LSTM->number_inputs * sizeof(T_));
        }
        
        // Propagate inputs into the LSTM.
        tmp_ptr_matrix_inputs[0u] = tmp_ptr_array_lstm_inputs;
        tmp_ptr_LSTM->Forward_Pass(1_zu, tmp_ptr_matrix_inputs);
        
        // Compute LSTM loss.
        {
            // Get outputs (financial data).
            memcpy(tmp_ptr_array_lstm_outputs,
                         tmp_ptr_Financial_Dataset->Get__Output_At(tmp_example_index),
                         tmp_number_recurrent_depth * tmp_ptr_Financial_Dataset->Get__Number_Outputs() * sizeof(T_));
            
            // Reset loss before proceeding to the computation of the loss.
            tmp_ptr_LSTM->Reset__Loss();
            
            // Compute the loss with l1 function.
            tmp_ptr_LSTM->Set__Loss_Function(MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L2);

            // Compute LSTM loss.
            tmp_ptr_matrix_outputs[0u] = tmp_ptr_array_lstm_outputs;
            tmp_ptr_LSTM->Compute__Loss(1_zu, tmp_ptr_matrix_outputs);

            // Increment accuracy trial.
            tmp_ptr_LSTM->number_accuracy_trial = tmp_ptr_Financial_Dataset->Get__Number_Outputs();

            // State of the trade.
            if(tmp_ptr_LSTM->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE) == 100_T)
            { ++tmp_count_wins; }
            else
            { ++tmp_count_lossses; }
            
            // Setup loss variable.
            tmp_ptr_LSTM->Set__Loss_Function(MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE);
            *tmp_ptr_LSTM->ptr_array_number_loss += tmp_example_index * tmp_ptr_Financial_Dataset->Get__Number_Outputs();
            tmp_summation_loss = *tmp_ptr_LSTM->ptr_array_loss_values += tmp_summation_loss;
            
            // Get loss.
            tmp_online_loss = tmp_ptr_LSTM->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);

            // Exponential moving average.
            if(tmp_example_index >= tmp_time_average && tmp_EMA != 0_T)
            {
                tmp_online_accuracy += tmp_EMA * (tmp_ptr_LSTM->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE) - tmp_online_accuracy);

                PRINT_FORMAT("T[%zu]: [L:%f | A:%f] --- [W:%zu vs L:%zu], Output[0]: %f." NEW_LINE,
                                         tmp_example_index,
                                         Cast_T(tmp_online_loss),
                                         Cast_T(tmp_online_accuracy),
                                         tmp_count_wins,
                                         tmp_count_lossses,
                                         Cast_T(tmp_ptr_LSTM->Get__Outputs(0_zu, tmp_number_recurrent_depth - 1_zu)[0u]));
            }
            // Moving average.
            else
            {
                tmp_online_accuracy += tmp_ptr_LSTM->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);

                PRINT_FORMAT("T[%zu]: [L:%f | A:%f] --- [W:%zu vs L:%zu], Output[0]: %f." NEW_LINE,
                                         tmp_example_index,
                                         Cast_T(tmp_online_loss),
                                         Cast_T(tmp_online_accuracy / static_cast<T_>(tmp_example_index + 1_zu)),
                                         tmp_count_wins,
                                         tmp_count_lossses,
                                         Cast_T(tmp_ptr_LSTM->Get__Outputs(0_zu, tmp_number_recurrent_depth - 1_zu)[0u]));
            }

            // Print output(s)
            for(int k(0); k != tmp_ptr_Financial_Dataset->Get__Number_Outputs(); ++k)
            {
                PRINT_FORMAT("Output[%d]: %f" NEW_LINE,
                                         k,
                                         Cast_T(tmp_ptr_LSTM->Get__Outputs(0_zu, tmp_number_recurrent_depth - 1_zu)[k]));
            }
        }
        // |END| Compute LSTM loss. |END|

        // Inverse preprocess SAEs.
        {
            // Price.
            // Price, Min-Max.
            for(tmp_index = 0_zu; tmp_index != 8_zu; ++tmp_index)
            {
                if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(tmp_index, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                
                if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(tmp_index, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            }
            
        #if defined(USE_MODWT)
            // Price, MODWT.
            for(tmp_index = 0_zu; tmp_index != 4_zu; ++tmp_index)
            {
                if(tmp_ptr_SAEs_Dataset->Preprocessing__MODWT_Inverse(tmp_index, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT_Inverse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                
                if(tmp_ptr_SAEs_Dataset->Preprocessing__MODWT_Inverse(tmp_index, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT_Inverse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            }
        #endif
            // |END| Price. |END|

            // RSI.
            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(9_zu, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(9_zu, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            // |END| RSI. |END|

            // ATR.
            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(10_zu, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(10_zu, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            // |END| ATR. |END|

            // StdDev.
            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(11_zu, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(tmp_ptr_SAEs_Dataset->Preprocessing__Minimum_Maximum_Inverse(11_zu, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            // |END| StdDev. |END|
        }
        // |END| Inverse preprocess SAEs. |END|
        
        // Append into the SAE dataset.
        if(tmp_append_to_dataset && tmp_SAE_Manager.Append_To_Dataset(tmp_ptr_array_financial_inputs, tmp_ptr_array_financial_inputs) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Append_To_Dataset()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        // Preprocess SAEs dataset.
        if(Preprocessing__SAEs(tmp_ptr_SAEs_Dataset) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__SAEs(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        // Datapoint loss/accuracy/output.
    #if defined(COMPILE_UI)
        if(tmp_optimization && tmp_print_chart)
        {
            tmp_ptr_SAEs_Dataset->Set__Plot__Output(tmp_example_index % 2_zu == 0_zu);

            tmp_ptr_LSTM_Dataset->Set__Plot__Output(tmp_example_index % 2_zu == 1_zu);
        }

        if(tmp_optimization)
        {
            tmp_ptr_SAEs_Dataset->Set__Plot__Loss(tmp_example_index % 2_zu == 0_zu);
            tmp_ptr_SAEs_Dataset->Set__Plot__Accuracy(tmp_example_index % 2_zu == 0_zu);

            tmp_ptr_LSTM_Dataset->Set__Plot__Loss(tmp_example_index % 2_zu == 1_zu);
            tmp_ptr_LSTM_Dataset->Set__Plot__Accuracy(tmp_example_index % 2_zu == 1_zu);
        }
        else
        {
            // Loss.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                        1u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                        static_cast<double>(tmp_example_index),
                                                                                                        tmp_online_loss);
            
            // Exponential moving average.
            if(tmp_example_index >= tmp_time_average && tmp_EMA != 0_T)
            {
                // Accuracy
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                            1u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_example_index),
                                                                                                            tmp_online_accuracy);
            }
            // Moving average.
            else
            {
                // Accuracy
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                            1u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_example_index),
                                                                                                            tmp_online_accuracy / static_cast<T_>(tmp_example_index + 1_zu));
            }
        }

        if(tmp_print_chart)
        {
            if(tmp_ptr_SAEs_Dataset->Plot__Dataset_Manager() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        else
        {
            // Desired output.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                        static_cast<double>(tmp_example_index),
                                                                                                        tmp_ptr_array_lstm_outputs[tmp_number_recurrent_depth - 1_zu]);

            // Predicted output.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                        1u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                        static_cast<double>(tmp_example_index),
                                                                                                        tmp_ptr_LSTM->Get__Outputs(0_zu, tmp_number_recurrent_depth - 1_zu)[0u]);
        }
    #endif
        
        // Optimization.
        if(tmp_optimization)
        {
            // Append into the LSTM dataset.
            if(tmp_append_to_dataset && tmp_LSTM_Manager.Append_To_Dataset(tmp_ptr_array_lstm_inputs, tmp_ptr_array_lstm_outputs) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Append_To_Dataset()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            
        #if defined(COMPILE_UI)
            // Reset the ploted loss.
            MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS);

            // Reset the ploted accuracy.
            MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY);

            // Plot the LSTM dataset.
            if(tmp_print_chart && tmp_ptr_LSTM_Dataset->Plot__Dataset_Manager() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        #endif
            
            // Set expiration SAEs optimization.
            tmp_while_condition.expiration = std::chrono::system_clock::now() + std::chrono::seconds(tmp_optimization_time_sae);
            if(tmp_SAE_Manager.Set__While_Condition_Optimization(tmp_while_condition) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__While_Condition_Optimization()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }
            
            // Testing the datapoint in the whole SAEs dataset.
            if(tmp_SAE_Manager.Testing_If_Require__Pre_Training() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Testing_If_Require__Pre_Training()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            
            // Optimize SAEs.
            tmp_thread_SAE = std::thread([&tmp_neural_network_manager = tmp_SAE_Manager]() { tmp_neural_network_manager.Pre_Training(); } );
            
            // Set expiration LSTM optimization.
            tmp_while_condition.expiration = std::chrono::system_clock::now() + std::chrono::seconds(tmp_optimization_time_lstm);
            if(tmp_LSTM_Manager.Set__While_Condition_Optimization(tmp_while_condition) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__While_Condition_Optimization()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }

            // Testing the datapoint in the whole LSTM dataset.
            if(tmp_LSTM_Manager.Testing_If_Require() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Testing_If_Require()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            // Optimize LSTM.
            tmp_thread_LSTM = std::thread(&MyEA::Neural_Network::Neural_Network_Manager::Optimization, &tmp_LSTM_Manager);
            
            auto tmp_Join_And_Compare([](std::thread &ref_thread_received, MyEA::Neural_Network::Neural_Network_Manager &ref_neural_network_manager_received) -> void
            {
                if(ref_thread_received.joinable())
                {
                    ref_thread_received.join();

                    ref_neural_network_manager_received.Compare_Trained();
                }
            });

            // Join the SAEs.
            if(tmp_thread_SAE.joinable())
            {
                tmp_thread_SAE.join();

                // SAE compare trained.
                if(tmp_SAE_Manager.Compare_Trained__Pre_Training())
                {
                    // Join and compare the LSTM.
                    tmp_Join_And_Compare(tmp_thread_LSTM, tmp_LSTM_Manager);

                    // Update the LSTM dataset.
                    if(tmp_ptr_LSTM_Dataset->Replace_Entries(tmp_ptr_SAEs_Dataset, tmp_ptr_SAE) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Replace_Entries(ptr, ptr)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                }
            }

            // Join and compare the LSTM.
            tmp_Join_And_Compare(tmp_thread_LSTM, tmp_LSTM_Manager);
        }

        // Convert moving average to exponential moving average.
        if(tmp_example_index + 1_zu == tmp_time_average && tmp_EMA != 0_T) { tmp_online_accuracy /= static_cast<T_>(tmp_example_index + 1_zu); }
    }
    
    delete[](tmp_ptr_array_lstm_outputs);
    delete[](tmp_ptr_array_lstm_inputs);
    delete[](tmp_ptr_array_sae_inputs);
    delete[](tmp_ptr_matrix_outputs);
    delete[](tmp_ptr_matrix_inputs);

#if defined(COMPILE_UI)
    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to print the SAEs chart?"))
    {
        tmp_ptr_SAEs_Dataset->Set__Plot__Output(true);

        if(tmp_ptr_SAEs_Dataset->Plot__Dataset_Manager() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to print the LSTM chart?"))
    {
        tmp_ptr_LSTM_Dataset->Set__Plot__Output(true);

        if(tmp_ptr_LSTM_Dataset->Plot__Dataset_Manager(ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    PAUSE_TERMINAL();
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: FORM: Deallocate." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Deallocate();
#endif

    return(true);
}
