#include "stdafx.hpp"

#include <Enums/Enum_Time_Frames.hpp>

#include <Tools/Message_Box.hpp>

#include <Neural_Network_APIv2.hpp>

// WARNING
#define SEQUENCE_WINDOW 96_zu

namespace MyEA
{
    namespace Neural_Network
    {
        bool Preprocessing__Post__Indicator(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received, class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            T_ tmp_minimum_input,
                 tmp_maximum_input;
            
            class Dataset<T_> *tmp_ptr_TrainingSet;
            
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_PRICE:
                    tmp_ptr_TrainingSet = ptr_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING);
                    
                    if(tmp_ptr_TrainingSet == nullptr) { return(false); }
                    
                    if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 6_zu)
                    {
                        // Zero centered.
                        //  Price.
                        for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                        {
                            tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                             tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                             tmp_index,
                                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                            tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                               tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                               tmp_index,
                                                                                                                               ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            tmp_index,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            tmp_index,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                        }
                        //  |END| Price. |END|
                        // |END| Zero centered. |END|
                        
                        // ATR.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                            4_zu,
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                            4_zu,
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                                4_zu,
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
                                                                                                                                4_zu,
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

                        // Momentum.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                            5_zu,
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                            5_zu,
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        5_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        5_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                        // |END| Momentum. |END|
                        
                        // Merge MODWT.
                        //  Price.
                        size_t tmp_J_level(MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum()));
                        size_t tmp_shift_index(0_zu);
                        
                        for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                        {
                            if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                tmp_J_level,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                                            MyEA::String::Get__Time().c_str(),
                                                            __FUNCTION__,
                                                            __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                tmp_J_level,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                                            MyEA::String::Get__Time().c_str(),
                                                            __FUNCTION__,
                                                            __LINE__);

                                return(false);
                            }
                            
                            tmp_shift_index += tmp_J_level;
                        }

                        tmp_J_level = MyEA::Math::Minimum<size_t>(5_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                        
                        for(size_t tmp_index(3_zu); tmp_index != 6_zu; ++tmp_index)
                        {
                            if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                tmp_J_level,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                tmp_J_level,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                            
                            tmp_shift_index += tmp_J_level;
                        }
                        //  |END| Price. |END|
                        
                        if(ptr_Dataset_Manager_received->Reallocate_Internal_Storage() == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate_Internal_Storage()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| Merge MODWT. |END|
                    }
                        break;
                default: break;
            }

            return(true);
        }

        std::vector<class Threaded_Neural_Network_Manager *> *global_ptr_vector_ptr_Threaded_Neural_Network_Manager = nullptr;
        
        Threaded_Neural_Network_Manager::Threaded_Neural_Network_Manager(bool const is_type_position_long_received, enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicator_received) : model_manager_client(is_type_position_long_received, type_indicator_received) { }

        class Threaded_Neural_Network_Manager* Get__Threaded_Neural_Network_Manager(std::vector<class Threaded_Neural_Network_Manager*> *const ptr_vector_Threaded_Neural_Network_Manager_received,
                                                                                                                                     bool const is_type_position_long_received,
                                                                                                                                     unsigned int const type_indicator_received)
        {
            if(ptr_vector_Threaded_Neural_Network_Manager_received == nullptr) { return(nullptr); }

            for(auto const &iterator: *ptr_vector_Threaded_Neural_Network_Manager_received)
            {
                if(iterator->model_manager_client.Get__Type_Indicator() == static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received)
                  &&
                  iterator->model_manager_client.Get__Is_Type_Position_Long() == is_type_position_long_received)
                { return(iterator); }
            }

            return(nullptr);
        }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Is_Loaded(void) { return(true); }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Initialize(bool const is_type_position_long_received,
                                                                                                     unsigned int const type_indicator_received,
                                                                                                     unsigned int const time_frames_received)
        {
            class Threaded_Neural_Network_Manager *tmp_ptr_Threaded_Neural_Network_Manager;

            if((tmp_ptr_Threaded_Neural_Network_Manager = new class Threaded_Neural_Network_Manager(is_type_position_long_received, static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received))) == nullptr) { return(false); }
            
            if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Initialize_Path(MyEA::Common::ENUM_TYPE_INDICATORS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received)], MyEA::Common::ENUM_TYPE_INDICATORS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received)] + "-[" + MyEA::Common::ENUM_TIME_FRAMES_NAMES[static_cast<enum MyEA::Common::ENUM_TIME_FRAMES>(time_frames_received)] + "]-" + (is_type_position_long_received ? "L" : "S")) == false)
            {
                SAFE_DELETE(tmp_ptr_Threaded_Neural_Network_Manager);

                return(false);
            }
            
            if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Connect() == false)
            {
                SAFE_DELETE(tmp_ptr_Threaded_Neural_Network_Manager);

                return(false);
            }
            
            tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Auto_Save_Dataset(false); // WARNING
            
            tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Optimization_Auto_Save_Trainer(true);

            tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Optimization_Auto_Save_Trained(true);

            global_ptr_vector_ptr_Threaded_Neural_Network_Manager->push_back(tmp_ptr_Threaded_Neural_Network_Manager);

            return(true);
        }
        
        DLL_EXTERNAL bool DLL_API API__Neural_Network__Initialize_Dataset_Manager(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Dataset_Manager() != nullptr) { return(true); }

                tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Comparison_Expiration(60_zu * 60_zu * 48_zu);

                struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;
                
                switch(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received))
                {
                    case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_PRICE:
                        tmp_Dataset_Manager_Parameters.type_storage = 2;
                        tmp_Dataset_Manager_Parameters.type_training = 1;

                        tmp_Dataset_Manager_Parameters.percent_training_size = 75.0;
                        tmp_Dataset_Manager_Parameters.percent_validation_size = 15.0;

                        tmp_Dataset_Manager_Parameters.training_parameters.value_0 = true;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_1 = 256;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_2 = 0;
                            break;
                }

                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false) { return(false); }

                class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Dataset_Manager());

                if(tmp_ptr_Dataset_Manager == nullptr) { return(false); }

                if(Preprocessing__Post__Indicator(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received), tmp_ptr_Dataset_Manager) == false) { return(false); }
            }
            else { return(false); }

            return(true);
        }

        DLL_EXTERNAL int DLL_API API__Neural_Network__Set__Output_Mode(bool const is_type_position_long_received,
                                                                                                                   unsigned int const type_indicator_received,
                                                                                                                   unsigned int const type_neural_network_use_received,
                                                                                                                   bool const use_last_layer_as_output_received)
        { return(true); }
        
        DLL_EXTERNAL int DLL_API API__Neural_Network__Set__Number_Inputs(bool const is_type_position_long_received,
                                                                                                                     unsigned int const type_indicator_received,
                                                                                                                     unsigned int const number_inputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Number_Inputs(number_inputs_received); }
            else { return(-1); }

            return(0);
        }

        DLL_EXTERNAL int DLL_API API__Neural_Network__Set__Number_Outputs(bool const is_type_position_long_received,
                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                        unsigned int const number_outputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Number_Outputs(number_outputs_received); }
            else { return(-1); }

            return(0);
        }
        
        DLL_EXTERNAL int DLL_API API__Neural_Network__Set__Number_Time_Predictions(bool const is_type_position_long_received,
                                                                                                                                     unsigned int const type_indicator_received,
                                                                                                                                     unsigned int const number_recurrent_depth_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Set__Number_Recurrent_Depth(number_recurrent_depth_received); }
            else { return(-1); }

            return(0);
        }
        
        DLL_EXTERNAL int DLL_API API__Neural_Network__Get__Is_Output_Symmetric(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        { return(true); }

        DLL_EXTERNAL int DLL_API API__Neural_Network__Get__Path_Neural_Network_Exist(bool const is_type_position_long_received,
                                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                                        unsigned int const type_neural_network_use_received)
        { return(true); }

        DLL_EXTERNAL int DLL_API API__Neural_Network__Get__Number_Inputs(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Number_Inputs(Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED))); }
            else { return(-1); }
        }

        DLL_EXTERNAL int DLL_API API__Neural_Network__Get__Number_Outputs(bool const is_type_position_long_received,
                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                        unsigned int const type_neural_network_use_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Number_Outputs(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)))); }
            else { return(-1); }
        }
        
        DLL_EXTERNAL int DLL_API API__Neural_Network__Get__Number_Time_Predictions(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Number_Recurrent_Depth(Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED))); }
            else { return(-1); }
        }
        
        DLL_EXTERNAL float DLL_API API__Neural_Network__Get__Loss(bool const is_type_position_long_received,
                                                                                                         unsigned int const type_indicator_received,
                                                                                                         unsigned int const type_neural_network_use_received,
                                                                                                         unsigned int const type_loss_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));
            
            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Send("loss");
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }
                
                std::string const tmp_buffer(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Receive());
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }
                
                return(std::stof(tmp_buffer));
            }
            else { return((std::numeric_limits<ST_>::max)()); }
        }
        
        DLL_EXTERNAL float DLL_API API__Neural_Network__Get__Accuracy(bool const is_type_position_long_received,
                                                                                                                unsigned int const type_indicator_received,
                                                                                                                unsigned int const type_neural_network_use_received,
                                                                                                                unsigned int const type_accuracy_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));
            
            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Send("accuarcy");
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }
                
                std::string const tmp_buffer(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Receive());
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }
                
                return(std::stof(tmp_buffer));
            }
            else { return((std::numeric_limits<ST_>::max)()); }
        }
        
        DLL_EXTERNAL T_ DLL_API API__Neural_Network__Get__Output(bool const is_type_position_long_received,
                                                                                                          unsigned int const type_indicator_received,
                                                                                                          unsigned int const type_neural_network_use_received,
                                                                                                          unsigned int const output_index_received,
                                                                                                          unsigned int const time_step_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));
            
            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                size_t const tmp_number_outputs(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Number_Outputs(Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED));
                size_t tmp_index;
                
                if(tmp_ptr_Threaded_Neural_Network_Manager->outputs.empty()) { return((std::numeric_limits<ST_>::max)()); }
                
                if(output_index_received >= tmp_number_outputs) { return((std::numeric_limits<ST_>::max)()); }
                
                if(time_step_received >= tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Number_Recurrent_Depth(Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED)) { return((std::numeric_limits<ST_>::max)()); }

                T_ *tmp_ptr_array_outputs(new T_[tmp_number_outputs]);
                T_ tmp_output;
                if(tmp_ptr_array_outputs == nullptr) { return((std::numeric_limits<ST_>::max)()); }
                
                std::istringstream tmp_output_stream(tmp_ptr_Threaded_Neural_Network_Manager->outputs);

                for(tmp_index = 0_zu; tmp_index != output_index_received + 1_zu; ++tmp_index)
                {
                    tmp_output_stream >> tmp_ptr_array_outputs[tmp_index];
                }

                tmp_output = tmp_ptr_array_outputs[output_index_received];

                delete[](tmp_ptr_array_outputs);

                return(tmp_output);
            }
            else { return((std::numeric_limits<ST_>::max)()); }
        }
        
        bool Preprocessing__Sequence_Window(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                                                    size_t const sequence_window_received,
                                                                    T_ *&ptr_array_inputs_received,
                                                                    class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_PRICE:
                    if(ptr_Dataset_Manager_received->Preprocessing__Sequence_Window(sequence_window_received,
                                                                                                                           1_zu,
                                                                                                                           ptr_array_inputs_received) == false)
                    {
                        delete[](ptr_array_inputs_received);

                        return(false);
                    }
                        break;
                default: break;
            }
            
            return(true);
        }
        
        bool Preprocessing__Pre(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                             enum ENUM_TYPE_INPUT const type_input_received,
                                             T_ *const ptr_array_source_inputs_received,
                                             T_ *&ptr_array_inputs_received,
                                             class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_PRICE:
                    if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 45_zu)
                    {
                        if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
                        {
                            size_t const tmp_old_input_size(6_zu),
                                               tmp_new_input_size(45_zu);
                            size_t tmp_J_level,
                                      tmp_shift_index,
                                      tmp_input_index;
                            
                            // Allocated merged inputs.
                            //  Initialize shifted index input position.
                            if((ptr_array_inputs_received = new T_[tmp_new_input_size]) == nullptr)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                            MyEA::String::Get__Time().c_str(),
                                                            __FUNCTION__,
                                                            tmp_new_input_size * sizeof(T_),
                                                            __LINE__);

                                return(false);
                            }
                            MEMSET(ptr_array_inputs_received,
                                            0,
                                            tmp_new_input_size * sizeof(T_));
                        
                            //  Convert small inputs dimensions into the large inputs dimensions.
                            tmp_shift_index = 0_zu;

                            tmp_J_level = MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                            for(tmp_input_index = 0_zu; tmp_input_index != 3_zu; ++tmp_input_index)
                            {
                                ptr_array_inputs_received[tmp_input_index + tmp_shift_index] = ptr_array_source_inputs_received[tmp_input_index];

                                tmp_shift_index += tmp_J_level;
                            }

                            tmp_J_level = MyEA::Math::Minimum<size_t>(5_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                            for(tmp_input_index = 3_zu; tmp_input_index != 6_zu; ++tmp_input_index)
                            {
                                ptr_array_inputs_received[tmp_input_index + tmp_shift_index] = ptr_array_source_inputs_received[tmp_input_index];

                                tmp_shift_index += tmp_J_level;
                            }
                            //  |END| Convert small inputs dimensions into the large inputs dimensions. |END|
                            // |END| Allocated merged inputs. |END|
                        
                            // Price.
                            tmp_shift_index = 0_zu;

                            tmp_J_level = MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());

                            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(tmp_index + tmp_shift_index,
                                                                                                                                ptr_array_inputs_received,
                                                                                                                                type_input_received) == false)
                                {
                                    delete[](ptr_array_inputs_received);

                                    return(false);
                                }

                                tmp_shift_index += tmp_J_level;
                            }
                            // |END| Price. |END|
                        
                            tmp_J_level = MyEA::Math::Minimum<size_t>(5_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());

                            // Shift RSI.
                            tmp_shift_index += tmp_J_level;

                            // ATR.
                            if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(4_zu + tmp_shift_index,
                                                                                                                                    ptr_array_inputs_received,
                                                                                                                                    type_input_received) == false)
                            {
                                delete[](ptr_array_inputs_received);

                                return(false);
                            }

                            tmp_shift_index += tmp_J_level;
                            // |END| ATR. |END|

                            // Momentum.
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(5_zu + tmp_shift_index,
                                                                                                                            ptr_array_inputs_received,
                                                                                                                            type_input_received) == false)
                            {
                                delete[](ptr_array_inputs_received);

                                return(false);
                            }
                            // |END| Momentum. |END|
                        
                            // Allocated merged inputs.
                            //  Convert large inputs dimensions into the small inputs dimensions.
                            tmp_shift_index = 0_zu;

                            tmp_J_level = MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                            for(tmp_input_index = 0_zu; tmp_input_index != 3_zu; ++tmp_input_index)
                            {
                                ptr_array_source_inputs_received[tmp_input_index] = ptr_array_inputs_received[tmp_input_index + tmp_shift_index];

                                tmp_shift_index += tmp_J_level;
                            }

                            tmp_J_level = MyEA::Math::Minimum<size_t>(5_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                            for(tmp_input_index = 3_zu; tmp_input_index != 6_zu; ++tmp_input_index)
                            {
                                ptr_array_source_inputs_received[tmp_input_index] = ptr_array_inputs_received[tmp_input_index + tmp_shift_index];

                                tmp_shift_index += tmp_J_level;
                            }
                            //  |END| Convert large inputs dimensions into the small inputs dimensions. |END|
                        
                            //  Change the large inputs dimensions into a small inputs dimensions.
                            delete[](ptr_array_inputs_received);

                            if((ptr_array_inputs_received = new T_[tmp_old_input_size]) == nullptr)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                            MyEA::String::Get__Time().c_str(),
                                                            __FUNCTION__,
                                                            tmp_old_input_size * sizeof(T_),
                                                            __LINE__);

                                return(false);
                            }
                        
                            MEMCPY(ptr_array_inputs_received,
                                            ptr_array_source_inputs_received,
                                            tmp_old_input_size * sizeof(T_));
                            //  |END| Change the large inputs dimensions into a small inputs dimensions. |END|
                            // |END| Allocated merged inputs. |END|
                        
                            // Merge MODWT.
                            //  Price, high, low, close.
                            tmp_shift_index = 0_zu;

                            tmp_J_level = MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                    tmp_old_input_size,
                                                                                                                                    ptr_array_inputs_received,
                                                                                                                                    type_input_received) == false)
                                {
                                    delete[](ptr_array_inputs_received);

                                    return(false);
                                }

                                tmp_shift_index += tmp_J_level;
                            }

                            tmp_J_level = MyEA::Math::Minimum<size_t>(5_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum());
                            for(size_t tmp_index(3_zu); tmp_index != 6_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                    tmp_old_input_size,
                                                                                                                                    ptr_array_inputs_received,
                                                                                                                                    type_input_received) == false)
                                {
                                    delete[](ptr_array_inputs_received);

                                    return(false);
                                }

                                tmp_shift_index += tmp_J_level;
                            }
                            //  |END| Price, high, low, close. |END|
                            // |END| Merge MODWT. |END|
                        }
                        else
                        {
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_array_source_inputs_received,
                                                                                                                            type_input_received) == false)
                            { return(false); }

                            ptr_array_inputs_received = ptr_array_source_inputs_received;
                        }
                    }
                        break;
                default: break;
            }
            
            return(true);
        }
        
        bool Preprocessing__Pre(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                             T_ *const ptr_array_source_inputs_received,
                                             T_ *const ptr_array_source_outputs_received,
                                             T_ *&ptr_array_inputs_received,
                                             T_ *&ptr_array_outputs_received,
                                             class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            if(ptr_array_source_inputs_received != nullptr)
            {
                if(Preprocessing__Pre(type_indicators_received,
                                                 ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                                 ptr_array_source_inputs_received,
                                                 ptr_array_inputs_received,
                                                 ptr_Dataset_Manager_received) == false) { return(false); }
            }
            
            if(ptr_array_source_outputs_received != nullptr)
            {
                if(Preprocessing__Pre(type_indicators_received,
                                                 ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                                 ptr_array_source_outputs_received,
                                                 ptr_array_outputs_received,
                                                 ptr_Dataset_Manager_received) == false) { return(false); }
            }

            return(true);
        }
        
        bool Preprocessing__Post(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                               T_ *&ptr_array_inputs_received,
                                               T_ *&ptr_array_outputs_received,
                                               class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_PRICE:
                    if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 45_zu)
                    {
                        SAFE_DELETE_ARRAY(ptr_array_inputs_received);
                        SAFE_DELETE_ARRAY(ptr_array_outputs_received);
                    }
                        break;
                default: break;
            }

            return(true);
        }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Forward_Pass(bool const is_type_position_long_received,
                                                                                                              unsigned int const type_indicator_received,
                                                                                                              unsigned int const type_neural_network_use_received,
                                                                                                              T_ *const ptr_array_inputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Dataset_Manager());

                if(tmp_ptr_Dataset_Manager == nullptr) { return(false); }

                size_t const tmp_number_inputs(tmp_ptr_Dataset_Manager->Get__Number_Inputs());
                size_t tmp_index;

                T_ *tmp_ptr_array_inputs(nullptr),
                     *tmp_ptr_array_outputs(nullptr);
                
                if(Preprocessing__Pre(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                ptr_array_inputs_received,
                                                nullptr,
                                                tmp_ptr_array_inputs,
                                                tmp_ptr_array_outputs,
                                                tmp_ptr_Dataset_Manager) == false)
                { return(false); }
                
                if(Preprocessing__Sequence_Window(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                                        SEQUENCE_WINDOW,
                                                                        tmp_ptr_array_inputs,
                                                                        tmp_ptr_Dataset_Manager) == false)
                { return(false); }

                tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Send("predict");
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }
                
                // Buffer.
                std::string tmp_buffer("");
                
                for(tmp_index = 0_zu; tmp_index != SEQUENCE_WINDOW * tmp_number_inputs - 1_zu; ++tmp_index)
                {
                    tmp_buffer += std::to_string(tmp_ptr_array_inputs[tmp_index]) + " ";
                }
                
                tmp_buffer += std::to_string(tmp_ptr_array_inputs[tmp_index]);
                // |END| Buffer. |END|

                tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Send(tmp_buffer);

                tmp_ptr_Threaded_Neural_Network_Manager->outputs = tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Receive();
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }

                if(Preprocessing__Post(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                  tmp_ptr_array_inputs,
                                                  tmp_ptr_array_outputs,
                                                  tmp_ptr_Dataset_Manager) == false)
                { return(false); }

                return(true);
            }
            else { return(false); }
        }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Append_To_Dataset_File(bool const is_type_position_long_received,
                                                                                                                              unsigned int const type_indicator_received,
                                                                                                                              T_ *ptr_array_inputs_received,
                                                                                                                              T_ *ptr_array_outputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Append_To_Dataset_History(ptr_array_inputs_received, ptr_array_outputs_received)); }
            else { return(false); }
        }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Append_To_Dataset(bool const is_type_position_long_received,
                                                                                                                      unsigned int const type_indicator_received,
                                                                                                                      T_ *const ptr_array_inputs_received,
                                                                                                                      T_ *const ptr_array_outputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Get__Dataset_Manager());

                if(tmp_ptr_Dataset_Manager == nullptr) { return(false); }
                
                T_ *tmp_ptr_array_inputs(nullptr),
                     *tmp_ptr_array_outputs(nullptr);
                
                if(Preprocessing__Pre(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                ptr_array_inputs_received,
                                                ptr_array_outputs_received,
                                                tmp_ptr_array_inputs,
                                                tmp_ptr_array_outputs,
                                                tmp_ptr_Dataset_Manager) == false) { return(false); }
                
                // Append to dataset.
                if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Append_To_Dataset(tmp_ptr_array_inputs, tmp_ptr_array_outputs) == false) { return(false); }
                
                if(Preprocessing__Post(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                  tmp_ptr_array_inputs,
                                                  tmp_ptr_array_outputs,
                                                  tmp_ptr_Dataset_Manager) == false) { return(false); }
                
                return(true);
            }
            else { return(false); }
        }
        
        DLL_EXTERNAL bool DLL_API API__Neural_Network__Join(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread.joinable())
                {
                    tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread.join();
                }

                return(true);
            }
            else { return(false); }
        }
        
        DLL_EXTERNAL bool DLL_API API__Neural_Network__Optimization(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread.joinable() == false)
                {
                    tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Send("optimize");
                    if(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Done() == false) { return(false); }

                    // Optimize and test if require.
                    tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread = std::thread(&MyEA::Neural_Network::Model_Manager_Client::Done, &tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client);
                }

                return(true);
            }
            else { return(false); }
        }
        
        DLL_EXTERNAL bool DLL_API API__Neural_Network__Write_File(bool const is_type_position_long_received,
                                                                                                        unsigned int const type_indicator_received,
                                                                                                        unsigned int const type_file_log_received,
                                                                                                        wchar_t const *const log_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                std::wstring const tmp_wchar_to_wstring(log_received);

                return(tmp_ptr_Threaded_Neural_Network_Manager->model_manager_client.Write_File(static_cast<enum MyEA::Common::ENUM_TYPE_FILE_LOG>(type_file_log_received), MyEA::String::Get__Time() + ": " + std::string(tmp_wchar_to_wstring.begin(), tmp_wchar_to_wstring.end())));
            }
            else { return(false); }
        }
        
        DLL_EXTERNAL bool DLL_API API__Neural_Network__Load_Neural_Network(bool const is_type_position_long_received,
                                                                                                                         unsigned int const type_indicator_received,
                                                                                                                         unsigned int const type_neural_network_use_received)
        { return(true); }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Save_Neural_Network(bool const is_type_position_long_received,
                                                                                                                         unsigned int const type_indicator_received,
                                                                                                                         unsigned int const type_neural_network_use_received)
        { return(true); }

        DLL_EXTERNAL bool DLL_API API__Neural_Network__Deinitialize(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            if(global_ptr_vector_ptr_Threaded_Neural_Network_Manager == nullptr) { return(false); }

            class Threaded_Neural_Network_Manager *tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                           is_type_position_long_received,
                                                                                                                                                                                                           type_indicator_received));
            
            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                // Find from vector.
                auto tmp_ptr_iteration_Threaded_Neural_Network_Manager(std::find(global_ptr_vector_ptr_Threaded_Neural_Network_Manager->begin(),
                                                                                                                   global_ptr_vector_ptr_Threaded_Neural_Network_Manager->end(),
                                                                                                                   tmp_ptr_Threaded_Neural_Network_Manager));

                // Remove from vector.
                if(tmp_ptr_iteration_Threaded_Neural_Network_Manager != global_ptr_vector_ptr_Threaded_Neural_Network_Manager->end()) { global_ptr_vector_ptr_Threaded_Neural_Network_Manager->erase(tmp_ptr_iteration_Threaded_Neural_Network_Manager); }

                // Deallocate.
                SAFE_DELETE(tmp_ptr_Threaded_Neural_Network_Manager);
            }

            if(global_ptr_vector_ptr_Threaded_Neural_Network_Manager->empty()) { SAFE_DELETE(global_ptr_vector_ptr_Threaded_Neural_Network_Manager); }

            return(true);
        }
        
        DLL_EXTERNAL bool DLL_API API__Neural_Network__Allocate(void)
        {
            if(global_ptr_vector_ptr_Threaded_Neural_Network_Manager == nullptr)
            {
                global_ptr_vector_ptr_Threaded_Neural_Network_Manager = new std::vector<class Threaded_Neural_Network_Manager *>();

                return(true);
            }
            else { return(false); }
        }
    }
}