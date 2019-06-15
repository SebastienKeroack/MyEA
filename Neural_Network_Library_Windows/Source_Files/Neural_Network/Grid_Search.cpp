#include "stdafx.hpp"

#if defined(COMPILE_UINPUT)
    #include <Tools/Key_Logger.hpp>
#endif

#if defined(COMPILE_UI)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <Form.hpp>
#endif

#include <Tools/Shutdown_Block.hpp>

#include <Neural_Network/Grid_Search.hpp>

#include <iostream>
#include <array>

template<typename T>
bool Dropout_Initializer__LW<T>::operator==(struct Dropout_Initializer__LW<T> const &ref_Dropout_Initializer__LW) const
{
    if(this == &ref_Dropout_Initializer__LW) { return(true); }

    return(this->layer_index == ref_Dropout_Initializer__LW.layer_index
             &&
             this->type_layer_dropout == ref_Dropout_Initializer__LW.type_layer_dropout
             &&
             this->value[0u] == ref_Dropout_Initializer__LW.value[0u]
             &&
             this->value[1u] == ref_Dropout_Initializer__LW.value[1u]);
}

template<typename T>
Grid_Search<T>::Grid_Search(void) { }

template<typename T>
Grid_Search<T>::~Grid_Search(void) { this->Deallocate__Stochastic_Index(); }

template<typename T>
void Grid_Search<T>::Shuffle(void)
{
    size_t tmp_swap,
              i;
    size_t tmp_randomize_index;
    
    for(i = this->_total_iterations; i--;)
    {
        this->_Generator_Random_Integer.Range(0, i);

        tmp_randomize_index = this->_Generator_Random_Integer.Generate_Integer();

        // Store the index to swap from the remaining index at "tmp_randomize_index"
        tmp_swap = this->_ptr_array_stochastic_index[tmp_randomize_index];

        // Get remaining index starting at index "i"
        // And store it to the remaining index at "tmp_randomize_index"
        this->_ptr_array_stochastic_index[tmp_randomize_index] = this->_ptr_array_stochastic_index[i];

        // Store the swapped index at the index "i"
        this->_ptr_array_stochastic_index[i] = tmp_swap;
    }
}

template<typename T>
void Grid_Search<T>::DataGridView_Initialize_Columns(void)
{
#if defined(COMPILE_UI)
    size_t tmp_vector_size,
                      tmp_vector_depth_index;
    
    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("Iteration");
    
    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("L-Training");

    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("L-Validating");

    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("L-Testing");

    if(this->p_vector_Weight_Decay.empty() == false) { MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("Weight decay"); }
    
    if(this->p_vector_Max_Norm_Constraints.empty() == false) { MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("Max-norm constraints"); }
    
    if(this->p_vector_Normalization_Momentum_Average.empty() == false) { MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("Normalization, momentum average"); }
    
    if(this->p_vector_layers_Dropout.empty() == false)
    {
        tmp_vector_size = this->p_vector_layers_Dropout.size();

        for(tmp_vector_depth_index = 0_zu; tmp_vector_depth_index != tmp_vector_size; ++tmp_vector_depth_index)
        {
            if(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).empty() == false)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Column("Dropout layer [" + std::to_string(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(0).layer_index) + "]");
            }
        }
    }
#endif
}

template<typename T>
void Grid_Search<T>::DataGridView_Add_Iteration(size_t const hyper_parameters_index_received, class Neural_Network *const ptr_Neural_Network_received)
{
#if defined(COMPILE_UI)
    size_t tmp_vector_size,
                      tmp_vector_depth_index;

    size_t tmp_depth_level(0),
              tmp_depth_level_shift(1),
              tmp_hyper_parameters_index(this->_use_shuffle ? this->_ptr_array_stochastic_index[hyper_parameters_index_received] : hyper_parameters_index_received),
              tmp_vector_hyper_parameters_index(tmp_hyper_parameters_index),
              tmp_cell_index(0);

    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(0u, std::to_string(hyper_parameters_index_received));
    
    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(1u, std::to_string(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
    
    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(2u, std::to_string(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
    
    MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(3u, std::to_string(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
    
    if(this->p_vector_Weight_Decay.empty() == false)
    {
        // Global index to vector index.
        tmp_vector_hyper_parameters_index = tmp_hyper_parameters_index;
        
        // Normalize.
        if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
        {
            tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

            tmp_depth_level_shift = tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);
        }

        // Increment depth.
        ++tmp_depth_level;

        // Overflow.
        if(tmp_vector_hyper_parameters_index >= this->p_vector_Weight_Decay.size()) { return; }
        
        MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(static_cast<unsigned int>(tmp_depth_level) + 3u, std::to_string(this->p_vector_Weight_Decay.at(tmp_vector_hyper_parameters_index)));
    }

    if(this->p_vector_Max_Norm_Constraints.empty() == false)
    {
        // Depth overflow.
        if(tmp_hyper_parameters_index < tmp_depth_level_shift + tmp_depth_level) { return; }

        // Global index to vector index.
        tmp_vector_hyper_parameters_index = tmp_hyper_parameters_index - (tmp_depth_level_shift + tmp_depth_level);

        // Normalize.
        if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
        {
            tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

            tmp_depth_level_shift += tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);
        }
        
        // Increment depth.
        ++tmp_depth_level;

        // Vector overflow.
        if(tmp_vector_hyper_parameters_index >= this->p_vector_Max_Norm_Constraints.size()) { return; }
        
        MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(static_cast<unsigned int>(tmp_depth_level) + 3u, std::to_string(this->p_vector_Max_Norm_Constraints.at(tmp_vector_hyper_parameters_index)));
    }
    
    if(this->p_vector_Normalization_Momentum_Average.empty() == false)
    {
        // Depth overflow.
        if(tmp_hyper_parameters_index < tmp_depth_level_shift + tmp_depth_level) { return; }

        // Global index to vector index.
        tmp_vector_hyper_parameters_index = tmp_hyper_parameters_index - (tmp_depth_level_shift + tmp_depth_level);

        // Normalize.
        if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
        {
            tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

            tmp_depth_level_shift += tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);
        }
        
        // Increment depth.
        ++tmp_depth_level;

        // Vector overflow.
        if(tmp_vector_hyper_parameters_index >= this->p_vector_Normalization_Momentum_Average.size()) { return; }
        
        MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(static_cast<unsigned int>(tmp_depth_level) + 3u, std::to_string(this->p_vector_Normalization_Momentum_Average.at(tmp_vector_hyper_parameters_index)));
    }
    
    if(this->p_vector_layers_Dropout.empty() == false)
    {
        tmp_vector_size = this->p_vector_layers_Dropout.size();

        for(tmp_vector_depth_index = 0_zu; tmp_vector_depth_index != tmp_vector_size; ++tmp_vector_depth_index)
        {
            if(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).empty() == false)
            {
                // Depth overflow.
                if(tmp_hyper_parameters_index < tmp_depth_level_shift + tmp_depth_level) { return; }

                // Global index to vector index.
                tmp_vector_hyper_parameters_index = tmp_hyper_parameters_index - (tmp_depth_level_shift + tmp_depth_level);

                // Normalize.
                if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
                {
                    tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

                    tmp_depth_level_shift += tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);
                }
                
                // Increment depth.
                ++tmp_depth_level;

                // Vector overflow.
                if(tmp_vector_hyper_parameters_index >= this->p_vector_layers_Dropout.at(tmp_vector_depth_index).size()) { return; }
                
                MyEA::Form::API__Form__Neural_Network__Chart_Grid_Search_Add_Row(static_cast<unsigned int>(tmp_depth_level) + 3u, std::string("Type: " + MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).type_layer_dropout] + ", Value[0]: " + std::to_string(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[0u]) + ", Value[1]: " + std::to_string(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[1u])));
            }
        }
    }
#endif
}

template<typename T>
bool Grid_Search<T>::Input(class MyEA::Neural_Network::Neural_Network_Manager &ref_Neural_Network_Manager)
{
    size_t tmp_layer_index;
    
    T tmp_minimum,
       tmp_maximum,
       tmp_step_size;
    
    class Neural_Network *const tmp_ptr_Neural_Network(ref_Neural_Network_Manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_COMPETITOR));
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    if(this->Set__Maximum_Iterations(MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Maximum iterations: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Iterations()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    if(this->Set__Use__Shuffle(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Use__Shuffle()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use weight decay: "))
    {
        tmp_minimum = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                         T(1),
                                                                                         MyEA::String::Get__Time() + ": Minimum: ");
        
        tmp_maximum = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                         T(1),
                                                                                         MyEA::String::Get__Time() + ": Maximum: ");

        tmp_step_size = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                         T(1),
                                                                                         MyEA::String::Get__Time() + ": Step size: ");

        if(this->Push_Back__Weight_Decay(tmp_minimum,
                                                             tmp_maximum,
                                                             tmp_step_size) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Weight_Decay(%f, %f, %f)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     Cast_T(tmp_minimum),
                                     Cast_T(tmp_maximum),
                                     Cast_T(tmp_step_size),
                                     __LINE__);

            return(false);
        }
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use max-norm constraints: "))
    {
        tmp_minimum = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Minimum: ");
        
        tmp_maximum = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Maximum: ");

        tmp_step_size = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Step size: ");

        if(this->Push_Back__Max_Norm_Constraints(tmp_minimum,
                                                                          tmp_maximum,
                                                                          tmp_step_size) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Max_Norm_Constraints(%f, %f, %f)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     Cast_T(tmp_minimum),
                                     Cast_T(tmp_maximum),
                                     Cast_T(tmp_step_size),
                                     __LINE__);

            return(false);
        }
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use batch-normalization momentum: "))
    {
        tmp_minimum = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                       T(1),
                                                                                       MyEA::String::Get__Time() + ": Minimum: ");
        
        tmp_maximum = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                       T(1),
                                                                                       MyEA::String::Get__Time() + ": Maximum: ");

        tmp_step_size = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                        T(1),
                                                                                        MyEA::String::Get__Time() + ": Step size: ");

        if(this->Push_Back__Normalization_Momentum_Average(tmp_minimum,
                                                                                           tmp_maximum,
                                                                                           tmp_step_size) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Normalization_Momentum_Average(%f, %f, %f)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     Cast_T(tmp_minimum),
                                     Cast_T(tmp_maximum),
                                     Cast_T(tmp_step_size),
                                     __LINE__);

            return(false);
        }

        if(this->Push_Back__Normalization_Momentum_Average(T(1) / static_cast<T>(ref_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Batch()),
                                                                                           T(1) / static_cast<T>(ref_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Batch()),
                                                                                           T(1),
                                                                                           false) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Normalization_Momentum_Average(%f, %f, 1.0, false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     Cast_T(T(1) / static_cast<T>(ref_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Batch())),
                                     Cast_T(T(1) / static_cast<T>(ref_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Batch())),
                                     __LINE__);

            return(false);
        }
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout: "))
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout, bernoulli: "))
        {
            if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the input layer: "))
            {
                struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                tmp_Dropout_Initializer__Arguments.layer_index = 0_zu;

                tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI;

                tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                         T(1),
                                                                                                                                                         MyEA::String::Get__Time() + ": Minimum: ");
                
                tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                          T(1),
                                                                                                                                                          MyEA::String::Get__Time() + ": Maximum: ");

                tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                 T(1),
                                                                                                                                                 MyEA::String::Get__Time() + ": Step size: ");
                
                if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Dropout_Initializer__Arguments.layer_index,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                             __LINE__);

                    return(false);
                }
            }

            for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_ptr_Neural_Network->Get__Total_Layers() - 1_zu; ++tmp_layer_index)
            {
                if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the hidden layer " + std::to_string(tmp_layer_index) + ": "))
                {
                    struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                    tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

                    tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI;

                    tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                             T(1),
                                                                                                                                                             MyEA::String::Get__Time() + ": Minimum: ");
                    
                    tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                              T(1),
                                                                                                                                                              MyEA::String::Get__Time() + ": Maximum: ");

                    tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                     T(1),
                                                                                                                                                     MyEA::String::Get__Time() + ": Step size: ");
                    
                    if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_Dropout_Initializer__Arguments.layer_index,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                                 __LINE__);

                        return(false);
                    }
                }
            }
        }
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout, bernoulli inverted: "))
        {
            if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the input layer: "))
            {
                struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                tmp_Dropout_Initializer__Arguments.layer_index = 0_zu;

                tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED;

                tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                         T(1),
                                                                                                                                                         MyEA::String::Get__Time() + ": Minimum: ");
                
                tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                          T(1),
                                                                                                                                                          MyEA::String::Get__Time() + ": Maximum: ");

                tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                 T(1),
                                                                                                                                                 MyEA::String::Get__Time() + ": Step size: ");
                
                if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Dropout_Initializer__Arguments.layer_index,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                             __LINE__);

                    return(false);
                }
            }

            for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_ptr_Neural_Network->Get__Total_Layers() - 1_zu; ++tmp_layer_index)
            {
                if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the hidden layer " + std::to_string(tmp_layer_index) + ": "))
                {
                    struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                    tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

                    tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED;

                    tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                             T(1),
                                                                                                                                                             MyEA::String::Get__Time() + ": Minimum: ");
                    
                    tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                              T(1),
                                                                                                                                                              MyEA::String::Get__Time() + ": Maximum: ");

                    tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                     T(1),
                                                                                                                                                     MyEA::String::Get__Time() + ": Step size: ");
                    
                    if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_Dropout_Initializer__Arguments.layer_index,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                                 __LINE__);

                        return(false);
                    }
                }
            }
        }
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout, gaussian: "))
        {
            if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the input layer: "))
            {
                struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                tmp_Dropout_Initializer__Arguments.layer_index = 0_zu;

                tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN;

                tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                         T(1),
                                                                                                                                                         MyEA::String::Get__Time() + ": Minimum: ");
                
                tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                          T(1),
                                                                                                                                                          MyEA::String::Get__Time() + ": Maximum: ");

                tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                 T(1),
                                                                                                                                                 MyEA::String::Get__Time() + ": Step size: ");
                
                if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Dropout_Initializer__Arguments.layer_index,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                             __LINE__);

                    return(false);
                }
            }

            for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_ptr_Neural_Network->Get__Total_Layers() - 1_zu; ++tmp_layer_index)
            {
                if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the hidden layer " + std::to_string(tmp_layer_index) + ": "))
                {
                    struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                    tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

                    tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN;

                    tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                             T(1),
                                                                                                                                                             MyEA::String::Get__Time() + ": Minimum: ");
                    
                    tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                              T(1),
                                                                                                                                                              MyEA::String::Get__Time() + ": Maximum: ");

                    tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                     T(1),
                                                                                                                                                     MyEA::String::Get__Time() + ": Step size: ");
                    
                    if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_Dropout_Initializer__Arguments.layer_index,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                                 __LINE__);

                        return(false);
                    }
                }
            }
        }
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout, Uout: "))
        {
            if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the input layer: "))
            {
                struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                tmp_Dropout_Initializer__Arguments.layer_index = 0_zu;

                tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT;

                tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                         T(1),
                                                                                                                                                         MyEA::String::Get__Time() + ": Minimum: ");
                
                tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                          T(1),
                                                                                                                                                          MyEA::String::Get__Time() + ": Maximum: ");

                tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                 T(1),
                                                                                                                                                 MyEA::String::Get__Time() + ": Step size: ");
                
                if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Dropout_Initializer__Arguments.layer_index,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                             __LINE__);

                    return(false);
                }
            }

            for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_ptr_Neural_Network->Get__Total_Layers() - 1_zu; ++tmp_layer_index)
            {
                if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the hidden layer " + std::to_string(tmp_layer_index) + ": "))
                {
                    struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                    tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

                    tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT;

                    tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                             T(1),
                                                                                                                                                             MyEA::String::Get__Time() + ": Minimum: ");
                    
                    tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                              T(1),
                                                                                                                                                              MyEA::String::Get__Time() + ": Maximum: ");

                    tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                     T(1),
                                                                                                                                                     MyEA::String::Get__Time() + ": Step size: ");
                    
                    if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_Dropout_Initializer__Arguments.layer_index,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                                 __LINE__);

                        return(false);
                    }
                }
            }
        }
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

        if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout, Zoneout: "))
        {
            if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the input layer: "))
            {
                struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                tmp_Dropout_Initializer__Arguments.layer_index = 0_zu;

                tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT;

                tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                           T(1),
                                                                                                                                                           MyEA::String::Get__Time() + ": Minimum_0: ");
                
                tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                            T(1),
                                                                                                                                                            MyEA::String::Get__Time() + ": Maximum_0: ");

                tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                   T(1),
                                                                                                                                                   MyEA::String::Get__Time() + ": Step size_0: ");
                
                tmp_Dropout_Initializer__Arguments.minimum_value[1u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                           T(1),
                                                                                                                                                           MyEA::String::Get__Time() + ": Minimum_1: ");
                
                tmp_Dropout_Initializer__Arguments.maximum_value[1u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                            T(1),
                                                                                                                                                            MyEA::String::Get__Time() + ": Maximum_1: ");

                tmp_Dropout_Initializer__Arguments.step_size[1u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                   T(1),
                                                                                                                                                   MyEA::String::Get__Time() + ": Step size_1: ");
                
                if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Dropout_Initializer__Arguments.layer_index,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[1u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[1u]),
                                             Cast_T(tmp_Dropout_Initializer__Arguments.step_size[1u]),
                                             __LINE__);

                    return(false);
                }
            }

            for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_ptr_Neural_Network->Get__Total_Layers() - 1_zu; ++tmp_layer_index)
            {
                if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use dropout at the hidden layer " + std::to_string(tmp_layer_index) + ": "))
                {
                    struct Dropout_Initializer__Arguments<T> tmp_Dropout_Initializer__Arguments;

                    tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

                    tmp_Dropout_Initializer__Arguments.type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT;

                    tmp_Dropout_Initializer__Arguments.minimum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                             T(1),
                                                                                                                                                             MyEA::String::Get__Time() + ": Minimum_0: ");
                    
                    tmp_Dropout_Initializer__Arguments.maximum_value[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                              T(1),
                                                                                                                                                              MyEA::String::Get__Time() + ": Maximum_0: ");

                    tmp_Dropout_Initializer__Arguments.step_size[0u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                     T(1),
                                                                                                                                                     MyEA::String::Get__Time() + ": Step size_0: ");
                    
                    tmp_Dropout_Initializer__Arguments.minimum_value[1u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                             T(1),
                                                                                                                                                             MyEA::String::Get__Time() + ": Minimum_1: ");
                    
                    tmp_Dropout_Initializer__Arguments.maximum_value[1u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                              T(1),
                                                                                                                                                              MyEA::String::Get__Time() + ": Maximum_1: ");

                    tmp_Dropout_Initializer__Arguments.step_size[1u] = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                                                                     T(1),
                                                                                                                                                     MyEA::String::Get__Time() + ": Step size_1: ");
                    
                    if(this->Push_Back__Dropout(tmp_Dropout_Initializer__Arguments) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back__Dropout(%zu, %s, %f, %f, %f, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_Dropout_Initializer__Arguments.layer_index,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_Dropout_Initializer__Arguments.type_layer_dropout].c_str(),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.step_size[0u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.minimum_value[1u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.maximum_value[1u]),
                                                 Cast_T(tmp_Dropout_Initializer__Arguments.step_size[1u]),
                                                 __LINE__);

                        return(false);
                    }
                }
            }
        }
    }
    
    return(true);
}

template<typename T>
bool Grid_Search<T>::Set__Maximum_Iterations(size_t const maximum_iterations_received)
{
    if(maximum_iterations_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum iterations can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    this->_maximum_iterations = maximum_iterations_received;

    return(true);
}

template<typename T>
bool Grid_Search<T>::Set__Use__Shuffle(bool const use_shuffle_received)
{
    this->_use_shuffle = use_shuffle_received;

    if(use_shuffle_received)
    {
        if(this->Allocate__Stochastic_Index() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Stochastic_Index()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        this->_Generator_Random_Integer.Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    }
    else { this->Deallocate__Stochastic_Index(); }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Push_Back__Weight_Decay(T const minimum_value_received,
                                                                           T const maxium_value_received,
                                                                           T const step_size_received,
                                                                           bool const allow_duplicate_received)
{
    if(minimum_value_received > maxium_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) bigger than maximum value (%f). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 Cast_T(maxium_value_received),
                                 __LINE__);

        return(false);
    }
    else if(minimum_value_received < T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 __LINE__);

        return(false);
    }
    else if(maxium_value_received > T(1))
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum value (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(maxium_value_received),
                                 __LINE__);

        return(false);
    }
    else if(step_size_received == T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Step size can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    T tmp_value;

    if(minimum_value_received == maxium_value_received)
    {
        tmp_value = minimum_value_received;

        this->Push_Back(this->p_vector_Weight_Decay,
                                  tmp_value,
                                  allow_duplicate_received);

        return(true);
    }

    size_t const tmp_size(static_cast<size_t>((maxium_value_received - minimum_value_received) / step_size_received) + 1u);
    size_t tmp_index;
    
    // Disable.
    if(minimum_value_received != T(0))
    {
        tmp_value = T(0);

        this->Push_Back(this->p_vector_Weight_Decay,
                                  tmp_value,
                                  allow_duplicate_received);
    }
    
    for(tmp_index = 0_zu; tmp_index != tmp_size; ++tmp_index)
    {
        tmp_value = minimum_value_received + static_cast<T>(tmp_index) * step_size_received;

        this->Push_Back(this->p_vector_Weight_Decay,
                                  tmp_value,
                                  allow_duplicate_received);
    }
    
    // Step size can not reach maximum.
    if(minimum_value_received + static_cast<T>(tmp_size - 1u) * step_size_received != maxium_value_received)
    {
        tmp_value = maxium_value_received;

        this->Push_Back(this->p_vector_Weight_Decay,
                                  tmp_value,
                                  allow_duplicate_received);
    }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Push_Back__Max_Norm_Constraints(T const minimum_value_received,
                                                                                        T const maxium_value_received,
                                                                                        T const step_size_received,
                                                                                        bool const allow_duplicate_received)
{
    if(minimum_value_received > maxium_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) bigger than maximum value (%f). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 Cast_T(maxium_value_received),
                                 __LINE__);

        return(false);
    }
    else if(minimum_value_received < T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 __LINE__);

        return(false);
    }
    else if(step_size_received == T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Step size can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    T tmp_value;
    
    if(minimum_value_received == maxium_value_received)
    {
        tmp_value = minimum_value_received;

        this->Push_Back(this->p_vector_Max_Norm_Constraints,
                                  tmp_value,
                                  allow_duplicate_received);

        return(true);
    }

    size_t const tmp_size(static_cast<size_t>((maxium_value_received - minimum_value_received) / step_size_received) + 1u);
    size_t tmp_index;
    
    // Disable.
    if(minimum_value_received != T(0))
    {
        tmp_value = T(0);

        this->Push_Back(this->p_vector_Max_Norm_Constraints,
                                  tmp_value,
                                  allow_duplicate_received);
    }
    
    for(tmp_index = 0_zu; tmp_index != tmp_size; ++tmp_index)
    {
        tmp_value = minimum_value_received + static_cast<T>(tmp_index) * step_size_received;

        this->Push_Back(this->p_vector_Max_Norm_Constraints,
                                  tmp_value,
                                  allow_duplicate_received);
    }
    
    // Step size can not reach maximum.
    if(minimum_value_received + static_cast<T>(tmp_size - 1u) * step_size_received != maxium_value_received)
    {
        tmp_value = maxium_value_received;

        this->Push_Back(this->p_vector_Max_Norm_Constraints,
                                  tmp_value,
                                  allow_duplicate_received);
    }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Push_Back__Normalization_Momentum_Average(T const minimum_value_received,
                                                                                                         T const maxium_value_received,
                                                                                                         T const step_size_received,
                                                                                                         bool const allow_duplicate_received)
{
    if(minimum_value_received > maxium_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) bigger than maximum value (%f). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 Cast_T(maxium_value_received),
                                 __LINE__);

        return(false);
    }
    else if(minimum_value_received < T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 __LINE__);

        return(false);
    }
    else if(maxium_value_received > T(1))
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum value (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(maxium_value_received),
                                 __LINE__);

        return(false);
    }
    else if(step_size_received == T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Step size can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    T tmp_value;
    
    if(minimum_value_received == maxium_value_received)
    {
        tmp_value = minimum_value_received;

        this->Push_Back(this->p_vector_Normalization_Momentum_Average,
                                  tmp_value,
                                  allow_duplicate_received);

        return(true);
    }

    size_t const tmp_size(static_cast<size_t>((maxium_value_received - minimum_value_received) / step_size_received) + 1u);
    size_t tmp_index;
    
    for(tmp_index = 0_zu; tmp_index != tmp_size; ++tmp_index)
    {
        tmp_value = minimum_value_received + static_cast<T>(tmp_index) * step_size_received;

        this->Push_Back(this->p_vector_Normalization_Momentum_Average,
                                  tmp_value,
                                  allow_duplicate_received);
    }
    
    // Step size can not reach maximum.
    if(minimum_value_received + static_cast<T>(tmp_size - 1u) * step_size_received != maxium_value_received)
    {
        tmp_value = maxium_value_received;

        this->Push_Back(this->p_vector_Normalization_Momentum_Average,
                                  tmp_value,
                                  allow_duplicate_received);
    }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Push_Back__Dropout(struct Dropout_Initializer__Arguments<T> Dropout_Initializer__Arguments_received, bool const allow_duplicate_received)
{
    if(Dropout_Initializer__Arguments_received.minimum_value[0u] > Dropout_Initializer__Arguments_received.maximum_value[0u])
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value[0u] (%f) bigger than maximum value[0u] (%f). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(Dropout_Initializer__Arguments_received.minimum_value[0u]),
                                 Cast_T(Dropout_Initializer__Arguments_received.maximum_value[0u]),
                                 __LINE__);

        return(false);
    }
    else if(Dropout_Initializer__Arguments_received.minimum_value[0u] < T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value[0u] (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(Dropout_Initializer__Arguments_received.minimum_value[0u]),
                                 __LINE__);

        return(false);
    }
    else if(Dropout_Initializer__Arguments_received.maximum_value[0u] > T(1))
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum value[0u] (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(Dropout_Initializer__Arguments_received.maximum_value[0u]),
                                 __LINE__);

        return(false);
    }
    else if(Dropout_Initializer__Arguments_received.step_size[0u] == T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Size_0 can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    switch(Dropout_Initializer__Arguments_received.type_layer_dropout)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
            if(Dropout_Initializer__Arguments_received.minimum_value[1u] > Dropout_Initializer__Arguments_received.maximum_value[1u])
            {
                PRINT_FORMAT("%s: %s: ERROR: Minimum value[1u] (%f) bigger than maximum value[1u] (%f). At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(Dropout_Initializer__Arguments_received.minimum_value[1u]),
                                         Cast_T(Dropout_Initializer__Arguments_received.maximum_value[1u]),
                                         __LINE__);

                return(false);
            }
            else if(Dropout_Initializer__Arguments_received.minimum_value[1u] < T(0))
            {
                PRINT_FORMAT("%s: %s: ERROR: Minimum value[1u] (%f) less than zero. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(Dropout_Initializer__Arguments_received.minimum_value[1u]),
                                         __LINE__);

                return(false);
            }
            else if(Dropout_Initializer__Arguments_received.maximum_value[1u] > T(1))
            {
                PRINT_FORMAT("%s: %s: ERROR: Maximum value[1u] (%f) bigger than one. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(Dropout_Initializer__Arguments_received.maximum_value[1u]),
                                         __LINE__);

                return(false);
            }
            else if(Dropout_Initializer__Arguments_received.step_size[1u] == T(0))
            {
                PRINT_FORMAT("%s: %s: ERROR: Size_1 can not be zero. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            if(Dropout_Initializer__Arguments_received.step_size[1u] >= T(1))
            {
                PRINT_FORMAT("%s: %s: ERROR: Size_1 can not be greater or equal to one. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
    }

    struct Dropout_Initializer__LW<T> tmp_Dropout_Initializer__LW;

    tmp_Dropout_Initializer__LW.layer_index = Dropout_Initializer__Arguments_received.layer_index;

    tmp_Dropout_Initializer__LW.type_layer_dropout = Dropout_Initializer__Arguments_received.type_layer_dropout;

    std::vector<struct Dropout_Initializer__LW<T>> *tmp_ptr_vector_Dropout(nullptr);
    
    // Search if a vector contains the desired layer.
    for(auto &ref_vector: this->p_vector_layers_Dropout)
    {
        if(ref_vector.at(0).layer_index == Dropout_Initializer__Arguments_received.layer_index)
        {
            tmp_ptr_vector_Dropout = &ref_vector;

            break; 
        }
    }

    // If the vector can not be find. Allocate a new one with the desired layer index.
    if(tmp_ptr_vector_Dropout == nullptr)
    {
        std::vector<struct Dropout_Initializer__LW<T>> tmp_vector;

        this->p_vector_layers_Dropout.push_back(tmp_vector);

        tmp_ptr_vector_Dropout = &this->p_vector_layers_Dropout.back();
    }
    
    if(Dropout_Initializer__Arguments_received.minimum_value[0u] == Dropout_Initializer__Arguments_received.maximum_value[0u]
       &&
       Dropout_Initializer__Arguments_received.minimum_value[1u] == Dropout_Initializer__Arguments_received.maximum_value[1u])
    {
        tmp_Dropout_Initializer__LW.value[0u] = Dropout_Initializer__Arguments_received.minimum_value[0u];

        tmp_Dropout_Initializer__LW.value[1u] = Dropout_Initializer__Arguments_received.minimum_value[1u];

        this->Push_Back(*tmp_ptr_vector_Dropout,
                                  tmp_Dropout_Initializer__LW,
                                  allow_duplicate_received);

        return(true);
    }
    
    size_t tmp_size,
              tmp_sub_size,
              tmp_index,
              tmp_sub_index;
    
    switch(Dropout_Initializer__Arguments_received.type_layer_dropout)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED: break;
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
            // Disable.
            if(Dropout_Initializer__Arguments_received.minimum_value[0u] != T(0) || Dropout_Initializer__Arguments_received.minimum_value[1u] != T(0))
            {
                tmp_Dropout_Initializer__LW.value[0u] = T(0);

                tmp_Dropout_Initializer__LW.value[1u] = T(0);
                
                this->Push_Back(*tmp_ptr_vector_Dropout,
                                          tmp_Dropout_Initializer__LW,
                                          false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type dropout (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        Dropout_Initializer__Arguments_received.type_layer_dropout,
                                        MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[Dropout_Initializer__Arguments_received.type_layer_dropout].c_str(),
                                        __LINE__);
            return(false);
    }
    
    // Value[0].
    if(Dropout_Initializer__Arguments_received.step_size[0u] != T(0))
    {
        tmp_size = static_cast<size_t>((Dropout_Initializer__Arguments_received.maximum_value[0u] - Dropout_Initializer__Arguments_received.minimum_value[0u]) / Dropout_Initializer__Arguments_received.step_size[0u]) + 1_zu;
        
        tmp_Dropout_Initializer__LW.value[1u] = T(0);

        for(tmp_index = 0_zu; tmp_index != tmp_size; ++tmp_index)
        {
            tmp_Dropout_Initializer__LW.value[0u] = Dropout_Initializer__Arguments_received.minimum_value[0u] + static_cast<T>(tmp_index) * Dropout_Initializer__Arguments_received.step_size[0u];

            this->Push_Back(*tmp_ptr_vector_Dropout,
                                      tmp_Dropout_Initializer__LW,
                                      allow_duplicate_received);
        }
    
        // Step size can not reach maximum.
        if(tmp_Dropout_Initializer__LW.value[0u] != Dropout_Initializer__Arguments_received.maximum_value[0u])
        {
            tmp_Dropout_Initializer__LW.value[0u] = Dropout_Initializer__Arguments_received.maximum_value[0u];
                
            this->Push_Back(*tmp_ptr_vector_Dropout,
                                        tmp_Dropout_Initializer__LW,
                                        false);
        }
    }
    // |END| Value[0]. |END|
    
    // Value[1].
    if(Dropout_Initializer__Arguments_received.step_size[1u] != T(0))
    {
        tmp_size = static_cast<size_t>((Dropout_Initializer__Arguments_received.maximum_value[1u] - Dropout_Initializer__Arguments_received.minimum_value[1u]) / Dropout_Initializer__Arguments_received.step_size[1u]) + 1_zu;
        
        tmp_Dropout_Initializer__LW.value[0u] = T(0);

        for(tmp_index = 0_zu; tmp_index != tmp_size; ++tmp_index)
        {
            tmp_Dropout_Initializer__LW.value[1u] = Dropout_Initializer__Arguments_received.minimum_value[1u] + static_cast<T>(tmp_index) * Dropout_Initializer__Arguments_received.step_size[1u];

            this->Push_Back(*tmp_ptr_vector_Dropout,
                                      tmp_Dropout_Initializer__LW,
                                      allow_duplicate_received);
        }
    
        // Step size can not reach maximum.
        if(tmp_Dropout_Initializer__LW.value[1u] != Dropout_Initializer__Arguments_received.maximum_value[1u])
        {
            tmp_Dropout_Initializer__LW.value[1u] = Dropout_Initializer__Arguments_received.maximum_value[1u];
                
            this->Push_Back(*tmp_ptr_vector_Dropout,
                                        tmp_Dropout_Initializer__LW,
                                        false);
        }
    }
    // |END| Value[1]. |END|
    
    // Value[0] && Value[1].
    if(Dropout_Initializer__Arguments_received.step_size[0u] * Dropout_Initializer__Arguments_received.step_size[1u] != T(0))
    {
        tmp_size = static_cast<size_t>((Dropout_Initializer__Arguments_received.maximum_value[0u] - Dropout_Initializer__Arguments_received.minimum_value[0u]) / Dropout_Initializer__Arguments_received.step_size[0u]) + 1_zu;
        tmp_sub_size = static_cast<size_t>((Dropout_Initializer__Arguments_received.maximum_value[1u] - Dropout_Initializer__Arguments_received.minimum_value[1u]) / Dropout_Initializer__Arguments_received.step_size[1u]) + 1_zu;
        
        for(tmp_index = 0_zu; tmp_index != tmp_size; ++tmp_index)
        {
            for(tmp_sub_index = 0_zu; tmp_sub_index != tmp_sub_size; ++tmp_sub_index)
            {
                tmp_Dropout_Initializer__LW.value[0u] = Dropout_Initializer__Arguments_received.minimum_value[0u] + static_cast<T>(tmp_index) * Dropout_Initializer__Arguments_received.step_size[0u];
                tmp_Dropout_Initializer__LW.value[1u] = Dropout_Initializer__Arguments_received.minimum_value[1u] + static_cast<T>(tmp_sub_index) * Dropout_Initializer__Arguments_received.step_size[1u];

                this->Push_Back(*tmp_ptr_vector_Dropout,
                                          tmp_Dropout_Initializer__LW,
                                          allow_duplicate_received);
            }
        }
    }
    // |END| Value[0] && Value[1]. |END|

    switch(Dropout_Initializer__Arguments_received.type_layer_dropout)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
            // Disable.
            if(Dropout_Initializer__Arguments_received.maximum_value[0u] != T(1))
            {
                tmp_Dropout_Initializer__LW.value[0u] = T(1);

                this->Push_Back(*tmp_ptr_vector_Dropout,
                                          tmp_Dropout_Initializer__LW,
                                          false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT: break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type dropout (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        Dropout_Initializer__Arguments_received.type_layer_dropout,
                                        MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[Dropout_Initializer__Arguments_received.type_layer_dropout].c_str(),
                                        __LINE__);
            return(false);
    }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Update_Tree(void)
{
    size_t tmp_vector_size,
                      tmp_vector_depth_index,
                      tmp_vector_depth_size;

    this->_vector_Tree.clear();

    tmp_vector_depth_size = 0_zu;

    if(this->p_vector_Weight_Decay.empty() == false)
    {
        this->_vector_Tree.push_back(1_zu);

        this->_vector_Tree.at(tmp_vector_depth_size) = this->p_vector_Weight_Decay.size();

        ++tmp_vector_depth_size;
    }
    
    if(this->p_vector_Max_Norm_Constraints.empty() == false)
    {
        this->_vector_Tree.push_back(1_zu);
        
        this->_vector_Tree.at(tmp_vector_depth_size) = this->p_vector_Max_Norm_Constraints.size();
        
        ++tmp_vector_depth_size;
    }

    if(this->p_vector_Normalization_Momentum_Average.empty() == false)
    {
        this->_vector_Tree.push_back(1_zu);
        
        this->_vector_Tree.at(tmp_vector_depth_size) = this->p_vector_Normalization_Momentum_Average.size();
        
        ++tmp_vector_depth_size;
    }
    
    if(this->p_vector_layers_Dropout.empty() == false)
    {
        tmp_vector_size = this->p_vector_layers_Dropout.size();

        for(tmp_vector_depth_index = 0_zu; tmp_vector_depth_index != tmp_vector_size; ++tmp_vector_depth_index)
        {
            if(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).empty() == false)
            {
                this->_vector_Tree.push_back(1_zu);
            
                this->_vector_Tree.at(tmp_vector_depth_size) = this->p_vector_layers_Dropout.at(tmp_vector_depth_index).size();
            
                ++tmp_vector_depth_size;
            }
        }
    }

    if(tmp_vector_depth_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Tree is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    size_t const tmp_trunc_size(this->_vector_Tree.at(0u));

    // Loop in each vectors.
    for(tmp_vector_depth_index = 0_zu; tmp_vector_depth_index != tmp_vector_depth_size - 1_zu; ++tmp_vector_depth_index)
    {
        // Self.
        this->_vector_Tree.at(tmp_vector_depth_index) = 1_zu;

        // Multiplacate path.
        this->_vector_Tree.at(tmp_vector_depth_index) += MyEA::Math::Recursive_Fused_Multiply_Add(this->_vector_Tree.data(),
                                                                                                                                                    tmp_vector_depth_index + 1_zu,
                                                                                                                                                    tmp_vector_depth_size - 1_zu);
    }
    
    // Last vector initialized at one.
    this->_vector_Tree.at(tmp_vector_depth_index) = 1_zu;

    if(this->_use_shuffle)
    {
        if(this->Reallocate__Stochastic_Index(tmp_trunc_size * this->_vector_Tree.at(0u)) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Stochastic_Index(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_trunc_size * this->_vector_Tree.at(0u),
                                     __LINE__);

            return(false);
        }
    }

    this->_total_iterations = tmp_trunc_size * this->_vector_Tree.at(0u);

    return(true);
}

template<typename T>
bool Grid_Search<T>::User_Controls(void)
{
#if defined(COMPILE_UINPUT)
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Maximum iterations (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_maximum_iterations);
        PRINT_FORMAT("%s:\t[1]: Use shuffle (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_use_shuffle ? "true" : "false");
        PRINT_FORMAT("%s:\t[2]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                2u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                if(this->Set__Maximum_Iterations(MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Maximum iterations: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Iterations()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                if(this->User_Controls__Shuffle() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Shuffle()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         2u,
                                         __LINE__);
                    break;
        }
    }
#endif

    return(true);
}

template<typename T>
bool Grid_Search<T>::User_Controls__Shuffle(void)
{
    bool const tmp_use_shuffle(this->_use_shuffle);

    if(this->Set__Use__Shuffle(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Use__Shuffle()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(tmp_use_shuffle == false && this->_use_shuffle) { this->Shuffle(); }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Search(class MyEA::Neural_Network::Neural_Network_Manager &ref_Neural_Network_Manager_received)
{
    if(this->_use_shuffle) { this->Shuffle(); }

    MyEA::Common::While_Condition tmp_While_Condition;

    tmp_While_Condition.type_while_condition = MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_ITERATION;

    tmp_While_Condition.maximum_iterations = this->_maximum_iterations;
    
    ref_Neural_Network_Manager_received.Set__While_Condition_Optimization(tmp_While_Condition);
    
    ref_Neural_Network_Manager_received.Set__Comparison_Expiration(0u);
    
    class Neural_Network *const tmp_ptr_Neural_Network_trainer(ref_Neural_Network_Manager_received.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER)),
                                    *const tmp_ptr_Neural_Network_competitor(ref_Neural_Network_Manager_received.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_COMPETITOR)),
                                    *tmp_ptr_Neural_Network_trainer_clone(nullptr);
    
    size_t tmp_hyper_parameters_iteration(0u);
    
#if defined(COMPILE_UINPUT)
    class Key_Logger tmp_Key_Logger;
#endif
    
    if((tmp_ptr_Neural_Network_trainer_clone = new class Neural_Network) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(class Neural_Network),
                                 __LINE__);

        return(false);
    }

    if(tmp_ptr_Neural_Network_trainer_clone->Copy(*tmp_ptr_Neural_Network_competitor, false) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(*ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        delete(tmp_ptr_Neural_Network_trainer_clone);

        return(false);
    }

    while(true)
    {
    #if defined(COMPILE_UINPUT)
    #if defined(COMPILE_WINDOWS)
        if(tmp_Key_Logger.Trigger_Key(0x45))
        {
            PRINT_FORMAT("%s: A signal for stopping the training has been triggered from the user input." NEW_LINE, MyEA::String::Get__Time().c_str());

            break;
        }
        else if(tmp_Key_Logger.Trigger_Key(0x50))
        {
            if(this->User_Controls() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
            }

            if(tmp_While_Condition.maximum_iterations != this->_maximum_iterations)
            {
                tmp_While_Condition.maximum_iterations = this->_maximum_iterations;

                if(ref_Neural_Network_Manager_received.Set__While_Condition_Optimization(tmp_While_Condition) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__While_Condition_Optimization(ref)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
            }
        }
    #elif defined(COMPILE_LINUX)
        tmp_Key_Logger.Collect_Keys_Pressed();

        if(tmp_Key_Logger.Trigger_Key('q'))
        {
            PRINT_FORMAT("%s: A signal for stopping the training has been triggered from the user input." NEW_LINE, MyEA::String::Get__Time().c_str());

            tmp_Key_Logger.Clear_Keys_Pressed();

            break;
        }
        else if(tmp_Key_Logger.Trigger_Key('m'))
        {
            tmp_Key_Logger.Clear_Keys_Pressed();
            
            if(this->User_Controls() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
            }

            if(tmp_While_Condition.maximum_iterations != this->_maximum_iterations)
            {
                tmp_While_Condition.maximum_iterations = this->_maximum_iterations;

                if(ref_Neural_Network_Manager_received.Set__While_Condition_Optimization(tmp_While_Condition) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__While_Condition_Optimization(ref)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
            }
        }
    #endif
    #endif

        if(tmp_ptr_Neural_Network_trainer->Update(*tmp_ptr_Neural_Network_trainer_clone, true) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(*ptr, true)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
            
            delete(tmp_ptr_Neural_Network_trainer_clone);

            return(false);
        }
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: [%zu]: Feed hyper parameters." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_hyper_parameters_iteration);
        if(this->Feed_Hyper_Parameters(this->_use_shuffle ? this->_ptr_array_stochastic_index[tmp_hyper_parameters_iteration] : tmp_hyper_parameters_iteration, tmp_ptr_Neural_Network_trainer) == false) { break; }
        
        if(tmp_ptr_Neural_Network_trainer->ptr_array_derivatives_parameters != nullptr) { tmp_ptr_Neural_Network_trainer->Clear_Training_Arrays(); }
        
        PRINT_FORMAT("%s: Search grid, optimization [%.2f%%] %zu / %zu." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 100.0 * static_cast<double>(tmp_hyper_parameters_iteration) / static_cast<double>(this->_total_iterations),
                                 tmp_hyper_parameters_iteration,
                                 this->_total_iterations);
        ref_Neural_Network_Manager_received.Optimization();
        
    #if defined(COMPILE_UI)
        // Trainer training datapoint.
        MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH,
                                                                                                    0u,
                                                                                                    MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                    static_cast<double>(tmp_hyper_parameters_iteration),
                                                                                                    tmp_ptr_Neural_Network_competitor->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
        
        // Trainer validating datapoint.
        MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH,
                                                                                                    0u,
                                                                                                    MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                    static_cast<double>(tmp_hyper_parameters_iteration),
                                                                                                    tmp_ptr_Neural_Network_competitor->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

        // Testing datapoint.
        if(ref_Neural_Network_Manager_received.Get__Dataset_Manager()->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
        {
            // Trainer testing datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                        static_cast<double>(tmp_hyper_parameters_iteration),
                                                                                                        tmp_ptr_Neural_Network_competitor->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
        }
        
        this->DataGridView_Add_Iteration(tmp_hyper_parameters_iteration, tmp_ptr_Neural_Network_competitor);
    #endif
        
        if(this->Get__On_Shutdown()) { break; }

        ++tmp_hyper_parameters_iteration;
    }
    
    delete(tmp_ptr_Neural_Network_trainer_clone);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Search grid, finish [%.2f%%] %zu / %zu." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             100.0 * static_cast<double>(tmp_hyper_parameters_iteration) / static_cast<double>(this->_total_iterations),
                             tmp_hyper_parameters_iteration,
                             this->_total_iterations);

    return(true);
}

template<typename T>
bool Grid_Search<T>::Feed_Hyper_Parameters(size_t const hyper_parameters_index_received, class Neural_Network *const ptr_Neural_Network_received)
{
    size_t tmp_vector_size,
                      tmp_vector_depth_index;

    size_t tmp_depth_level(0),
              tmp_depth_level_shift(1),
              tmp_vector_hyper_parameters_index(hyper_parameters_index_received);

    if(this->p_vector_Weight_Decay.empty() == false)
    {
        // Global index to vector index.
        tmp_vector_hyper_parameters_index = hyper_parameters_index_received;
        
        // Normalize.
        if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
        {
            tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

            tmp_depth_level_shift = tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);

            ++tmp_depth_level;
        }

        // Overflow.
        if(tmp_vector_hyper_parameters_index >= this->p_vector_Weight_Decay.size()) { return(false); }
        
        PRINT_FORMAT("%s: [%zu]: Weight_Decay(%zu): %f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 hyper_parameters_index_received,
                                 tmp_vector_hyper_parameters_index,
                                 Cast_T(this->p_vector_Weight_Decay.at(tmp_vector_hyper_parameters_index)));
        if(ptr_Neural_Network_received->Set__Regularization__Weight_Decay(this->p_vector_Weight_Decay.at(tmp_vector_hyper_parameters_index)) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Weight_Decay(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(this->p_vector_Weight_Decay.at(tmp_vector_hyper_parameters_index)),
                                         __LINE__);

                return(false);
            }
    }

    if(this->p_vector_Max_Norm_Constraints.empty() == false)
    {
        // Depth overflow.
        if(hyper_parameters_index_received < tmp_depth_level_shift + tmp_depth_level) { return(true); }

        // Global index to vector index.
        tmp_vector_hyper_parameters_index = hyper_parameters_index_received - (tmp_depth_level_shift + tmp_depth_level);

        // Normalize.
        if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
        {
            tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

            tmp_depth_level_shift += tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);

            ++tmp_depth_level;
        }

        // Vector overflow.
        if(tmp_vector_hyper_parameters_index >= this->p_vector_Max_Norm_Constraints.size()) { return(false); }
        
        PRINT_FORMAT("%s: [%zu]: Max_Norm_Constraints(%zu): %f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 hyper_parameters_index_received,
                                 tmp_vector_hyper_parameters_index,
                                 Cast_T(this->p_vector_Max_Norm_Constraints.at(tmp_vector_hyper_parameters_index)));
        if(ptr_Neural_Network_received->Set__Regularization__Max_Norm_Constraints(this->p_vector_Max_Norm_Constraints.at(tmp_vector_hyper_parameters_index)) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Max_Norm_Constraints(%f)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     Cast_T(this->p_vector_Max_Norm_Constraints.at(tmp_vector_hyper_parameters_index)),
                                     __LINE__);

            return(false);
        }
    }
    
    if(this->p_vector_Normalization_Momentum_Average.empty() == false)
    {
        // Depth overflow.
        if(hyper_parameters_index_received < tmp_depth_level_shift + tmp_depth_level) { return(true); }

        // Global index to vector index.
        tmp_vector_hyper_parameters_index = hyper_parameters_index_received - (tmp_depth_level_shift + tmp_depth_level);

        // Normalize.
        if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
        {
            tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

            tmp_depth_level_shift += tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);

            ++tmp_depth_level;
        }

        // Vector overflow.
        if(tmp_vector_hyper_parameters_index >= this->p_vector_Normalization_Momentum_Average.size()) { return(false); }
        
        PRINT_FORMAT("%s: [%zu]: Normalization_Momentum_Average(%zu): %f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 hyper_parameters_index_received,
                                 tmp_vector_hyper_parameters_index,
                                 Cast_T(this->p_vector_Normalization_Momentum_Average.at(tmp_vector_hyper_parameters_index)));
        if(ptr_Neural_Network_received->Set__Normalization_Momentum_Average(this->p_vector_Normalization_Momentum_Average.at(tmp_vector_hyper_parameters_index)) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Momentum_Average(%f)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     Cast_T(this->p_vector_Normalization_Momentum_Average.at(tmp_vector_hyper_parameters_index)),
                                     __LINE__);

            return(false);
        }
    }
    
    if(this->p_vector_layers_Dropout.empty() == false)
    {
        tmp_vector_size = this->p_vector_layers_Dropout.size();

        for(tmp_vector_depth_index = 0_zu; tmp_vector_depth_index != tmp_vector_size; ++tmp_vector_depth_index)
        {
            if(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).empty() == false)
            {
                // Depth overflow.
                if(hyper_parameters_index_received < tmp_depth_level_shift + tmp_depth_level) { return(true); }

                // Global index to vector index.
                tmp_vector_hyper_parameters_index = hyper_parameters_index_received - (tmp_depth_level_shift + tmp_depth_level);

                // Normalize.
                if(tmp_depth_level + 1_zu != this->_vector_Tree.size())
                {
                    tmp_vector_hyper_parameters_index /= this->_vector_Tree.at(tmp_depth_level);

                    tmp_depth_level_shift += tmp_vector_hyper_parameters_index * this->_vector_Tree.at(tmp_depth_level);

                    ++tmp_depth_level;
                }

                // Vector overflow.
                if(tmp_vector_hyper_parameters_index >= this->p_vector_layers_Dropout.at(tmp_vector_depth_index).size()) { return(false); }
            
                PRINT_FORMAT("%s: [%zu]: Dropout(%zu): Layer(%zu), Type(%u | %s), Value[0](%f), Value[1](%f)" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         hyper_parameters_index_received,
                                         tmp_vector_hyper_parameters_index,
                                         this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).layer_index,
                                         this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).type_layer_dropout,
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).type_layer_dropout].c_str(),
                                         Cast_T(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[0u]),
                                         Cast_T(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[1u]));
                if(ptr_Neural_Network_received->Set__Dropout(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).layer_index,
                                                                                    this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).type_layer_dropout,
                                                                                    std::array<T, 2_zu>{this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[0u], this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[1u]}.data(),
                                                                                    true) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u | %s, %f, %f, true)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).layer_index,
                                             this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).type_layer_dropout,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).type_layer_dropout].c_str(),
                                             Cast_T(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[0u]),
                                             Cast_T(this->p_vector_layers_Dropout.at(tmp_vector_depth_index).at(tmp_vector_hyper_parameters_index).value[1u]),
                                             __LINE__);

                    return(false);
                }
            }
        }
    }

    return(tmp_depth_level != 0_zu);
}

template<typename T>
bool Grid_Search<T>::Allocate__Shutdown_Boolean(void)
{
    std::atomic<bool> *tmp_ptr_shutdown_boolean(new std::atomic<bool>);

    if(tmp_ptr_shutdown_boolean == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(std::atomic<bool>),
                                 __LINE__);

        return(false);
    }

    this->_ptr_shutdown_boolean = tmp_ptr_shutdown_boolean;
    
    this->_ptr_shutdown_boolean->store(false);

    return(true);
}

template<typename T>
bool Grid_Search<T>::Assign_Shutdown_Block(class Shutdown_Block &ref_Shutdown_Block_received)
{
    if(this->Allocate__Shutdown_Boolean() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Shutdown_Boolean()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                
        return(false);
    }
    else if(ref_Shutdown_Block_received.Push_Back(this->_ptr_shutdown_boolean) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Grid_Search<T>::Get__On_Shutdown(void) const { return(this->_ptr_shutdown_boolean != nullptr && this->_ptr_shutdown_boolean->load()); }

template<typename T>
template<typename O> void Grid_Search<T>::Push_Back(std::vector<O> &ref_vector_received,
                                                                                      O &ref_object_received,
                                                                                      bool const allow_duplicate_received)
{
    if(allow_duplicate_received) { ref_vector_received.push_back(ref_object_received); }
    else if(std::find(ref_vector_received.begin(),
                          ref_vector_received.end(),
                          ref_object_received) == ref_vector_received.end())
    { ref_vector_received.push_back(ref_object_received); }
}

template<typename T>
void Grid_Search<T>::Deallocate__Stochastic_Index(void) { SAFE_DELETE(this->_ptr_array_stochastic_index); }

template<typename T>
bool Grid_Search<T>::Allocate__Stochastic_Index(void)
{
    size_t const tmp_stochastic_index_size(this->_total_iterations == 0_zu ? 1_zu : this->_total_iterations);

    this->_ptr_array_stochastic_index = new size_t[tmp_stochastic_index_size];
    if(this->_ptr_array_stochastic_index == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_stochastic_index_size * sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    for(size_t tmp_index(0u); tmp_index != tmp_stochastic_index_size; ++tmp_index)
    { this->_ptr_array_stochastic_index[tmp_index] = tmp_index; }
    
    return(true);
}

template<typename T>
bool Grid_Search<T>::Reallocate__Stochastic_Index(size_t const total_iterations_received)
{
    this->_ptr_array_stochastic_index = Memory::reallocate_cpp<size_t>(this->_ptr_array_stochastic_index,
                                                                                                          total_iterations_received,
                                                                                                          this->_total_iterations,
                                                                                                          false);
    if(this->_ptr_array_stochastic_index == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 total_iterations_received * sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    for(size_t tmp_index(0u); tmp_index != total_iterations_received; ++tmp_index)
    { this->_ptr_array_stochastic_index[tmp_index] = tmp_index; }
    
    return(true);
}

// template struct/class initialization declaration.
template struct Dropout_Initializer__LW<T_>;
template class Grid_Search<T_>;

// template function initialization declaration.
template void Grid_Search<T_>::Push_Back(std::vector<struct Dropout_Initializer__LW<T_>> &ref_vector_received,
                                                                 struct Dropout_Initializer__LW<T_> &ref_object_received,
                                                                 bool const allow_duplicate_received);
template void Grid_Search<T_>::Push_Back(std::vector<T_> &ref_vector_received,
                                                                 T_ &ref_object_received,
                                                                 bool const allow_duplicate_received);