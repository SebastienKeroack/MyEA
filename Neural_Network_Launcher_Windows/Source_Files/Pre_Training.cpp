#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <windows.h>
    #include <Form.hpp>
#endif

#include <Pre_Training.hpp>

#include <Tools/Shutdown_Block.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Pre_Training(class Shutdown_Block &ref_Shutdown_Block_received)
{
    std::string tmp_neural_network_name;
    
    std::cout << MyEA::String::Get__Time() << ": Neural network name: ";

    getline(std::cin, tmp_neural_network_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_neural_network_name + " Pre-Training - Neural Network").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE);
    
    tmp_Neural_Network_Manager.Set__Optimization_Auto_Save_Trainer(true);

    tmp_Neural_Network_Manager.Set__Optimization_Auto_Save_Competitor(false);
    
    tmp_Neural_Network_Manager.Set__Optimization_Auto_Save_Trained(true);

    if(tmp_Neural_Network_Manager.Set__Desired_Loss(MyEA::String::Cin_Real_Number<T_>(0.0,
                                                                                                                                         1.0,
                                                                                                                                         MyEA::String::Get__Time() + ": Desired loss: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Loss()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Initialize_Path(tmp_neural_network_name, tmp_neural_network_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_neural_network_name.c_str(),
                                 tmp_neural_network_name.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Initialize_Dataset_Manager() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    // Memory allocate.
    size_t const tmp_remaining_available_system_memory(Get__Remaining_Available_System_Memory(10.0L, KILOBYTE * KILOBYTE * KILOBYTE));

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Maximum available memory allocatable:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu] MBs." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE);

    size_t const tmp_maximum_host_memory_allocate_bytes(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                               tmp_remaining_available_system_memory / KILOBYTE / KILOBYTE,
                                                                                                                                               MyEA::String::Get__Time() + ": Maximum memory allocation (MBs): ") * 1024u * 1024u);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    // |END| Memory allocate. |END|

    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to load the neural network from a file?"))
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
    }
    else
    {
        if(tmp_Neural_Network_Manager.Create_Neural_Network(tmp_maximum_host_memory_allocate_bytes) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Create_Neural_Network()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    class Neural_Network *const tmp_ptr_neural_network(tmp_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER));

    if(tmp_ptr_neural_network == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural network is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    std::vector<size_t> tmp_vector_epochs_per_pre_training_level;
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Number of epochs per pre-training level." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=250'000." NEW_LINE, MyEA::String::Get__Time().c_str());

    for(size_t tmp_level(1_zu); tmp_level != (tmp_ptr_neural_network->total_layers - 3_zu) / 2_zu + 2_zu; ++tmp_level)
    { tmp_vector_epochs_per_pre_training_level.push_back(MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Maximum epochs, level " + std::to_string(tmp_level) + ": ")); }

#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: FORM: Allocate." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Allocate();

    PRINT_FORMAT("%s: FORM: Initialize chart of type loss with 2 series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS, 2u);

    PRINT_FORMAT("%s: FORM: Initialize chart of type accuracy with 2 series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY, 2u);
    
    PRINT_FORMAT("%s: FORM: Initialize chart of type output with 2 series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT, 2u);
    
    PRINT_FORMAT("%s: FORM: Initialize chart of type gradient with 1 serie." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH, 1u);

    PRINT_FORMAT("%s: FORM: Enable training series." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Use_Datapoint_Training(true);

    PRINT_FORMAT("%s: FORM: Initialize chart means at 100 datapoint." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Total_Means(100u);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    bool const tmp_dataset_plot(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to plot the dataset?"));

    PRINT_FORMAT("%s: FORM: Set dataset plot to %s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             tmp_dataset_plot ? "true" : "false");
    tmp_Neural_Network_Manager.Get__Dataset_Manager()->Set__Plot__Output(tmp_dataset_plot);
    
    if(tmp_dataset_plot)
    {
        PRINT_FORMAT("%s: FORM: Maximum ploted example(s)." NEW_LINE, MyEA::String::Get__Time().c_str());
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->User_Controls__Set__Maximum_Ploted_Example() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Maximum_Ploted_Example()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        PRINT_FORMAT("%s: FORM: Time delay ploted." NEW_LINE, MyEA::String::Get__Time().c_str());
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->User_Controls__Set__Time_Delay_Ploted() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Time_Delay_Ploted()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
#endif // COMPILE_UI
    
    if(ref_Shutdown_Block_received.Create_Shutdown_Block(true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Create_Shutdown_Block(true)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Assign_Shutdown_Block(ref_Shutdown_Block_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign_Shutdown_Block(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(ref_Shutdown_Block_received.Get__On_Shutdown() == false
          &&
          ref_Shutdown_Block_received.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Assign_Shutdown_Block(ref_Shutdown_Block_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign_Shutdown_Block(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(ref_Shutdown_Block_received.Get__On_Shutdown() == false
          &&
          ref_Shutdown_Block_received.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(ref_Shutdown_Block_received.Peak_Message_Async() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Peak_Message_Async()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        if(ref_Shutdown_Block_received.Get__On_Shutdown() == false
          &&
          ref_Shutdown_Block_received.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    
    if(tmp_Neural_Network_Manager.Testing__Pre_Training() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Testing__Pre_Training()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(tmp_Neural_Network_Manager.Pre_Training(tmp_vector_epochs_per_pre_training_level) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Pre_Training(vector[%zu])\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_vector_epochs_per_pre_training_level.size(),
                                 __LINE__);
        
        if(ref_Shutdown_Block_received.Get__On_Shutdown() == false
          &&
          ref_Shutdown_Block_received.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    
    tmp_Neural_Network_Manager.Compare_Trained__Pre_Training();

    if(tmp_Neural_Network_Manager.Save_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_Neural_Network(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL].c_str(),
                                 __LINE__);
        
        if(ref_Shutdown_Block_received.Get__On_Shutdown() == false
          &&
          ref_Shutdown_Block_received.Remove_Shutdown_Block() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    
#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: FORM: Deallocate." NEW_LINE, MyEA::String::Get__Time().c_str());
    MyEA::Form::API__Form__Neural_Network__Deallocate();
#endif

    if(ref_Shutdown_Block_received.Get__On_Shutdown() == false
      &&
      ref_Shutdown_Block_received.Remove_Shutdown_Block() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Remove_Shutdown_Block()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}
