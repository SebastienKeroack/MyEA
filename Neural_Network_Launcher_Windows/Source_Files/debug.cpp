#include "stdafx.hpp"
#include "main.hpp"
#include "debug.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif // COMPILE_WINDOWS

#include <Files/File.hpp>

#if defined(COMPILE_CUDA)
    #include <CUDA/CUDA_Dataset_Manager.cuh>
#endif // COMPILE_CUDA

#include <csignal>
#include <limits>
#include <chrono>
#include <iostream>
#include <array>

void simple_debug(bool const nsight_received)
{
#if defined(COMPILE_WINDOWS)
    SetConsoleTitle("Debug - Neural Network");
#endif // COMPILE_WINDOWS

#if defined(COMPILE_CUDA)
    int const tmp_cuda_device_index(0);

    bool const tmp_use_CUDA(false);

    size_t tmp_memory_allocate_device(0_zu);

    struct cudaDeviceProp tmp_cuda_device;
#endif // COMPILE_CUDA
    
    bool tmp_use_OpenMP(false);
    bool const tmp_simulate_online_training(false),
               tmp_load_parameters(false),
               tmp_use_dropout(false),
               tmp_use_normalization(true),
               tmp_use_layer_normalization_before_activation(true),
               tmp_use_warm_restarts(false),
               tmp_use_bidirectional(false),
               tmp_use_residual(true),
               tmp_use_pooling(true && tmp_use_residual),
               tmp_use_bottleneck(false && tmp_use_residual),
               tmp_pooling_ceil_mode(false),
               tmp_clip_gradient(false),
               tmp_tied_parameter(false),
               tmp_save(false),
               tmp_copy(true),
               tmp_use_training(true),
               tmp_use_update_bn(true && tmp_use_normalization),
               tmp_use_validating(false),
               tmp_use_testing(true),
               tmp_use_adept(false),
               tmp_use_testing_print_input(false),
               tmp_use_testing_print_output(false),
               tmp_print_parameters(false);

    unsigned int const tmp_weights_seed(5413u);

    long long int tmp_widening_factor_alphas[2u] = {-9ll, 0ll};
    
    size_t const tmp_number_layers(12_zu),
                 tmp_number_units[2u] = {32_zu, 1_zu},
                 tmp_residual_block_width(3_zu),
                 tmp_pooling_kernel_size(2_zu),
                 tmp_pooling_stride(2_zu),
                 tmp_pooling_padding(0_zu),
                 tmp_pooling_dilation(0_zu),
                 tmp_time_delays(27_zu),
                 tmp_epochs(10_zu),
                 tmp_sub_epochs(10_zu),
                 tmp_length_run(nsight_received ? 1_zu : 1_zu),
                 tmp_image_size(28_zu),
                 tmp_memory_allocate(256_zu * KILOBYTE * KILOBYTE);

    T_ const tmp_dropout_values[2u] = {0.5_T, 0.5_T},
             tmp_regularization_l1(0_T),
             tmp_regularization_l2(0_T),
             tmp_regularization_max_norm_constraints(8_T),
             tmp_regularization_weight_decay(0_T),
             tmp_accuracy_variance(0.49_T),
             tmp_activation_steepness(1_T),
             tmp_learning_rate(1e-3_T),
             tmp_learning_rate_final(1e-1_T),
             tmp_weights_minimum(-1_T),
             tmp_weights_maximum(1_T),
             tmp_clip_gradient_value(1_T),
             tmp_warm_restarts_minimum_learning_rate(1e-9_T),
             tmp_warm_restarts_maximum_learning_rate(0.01_T),
             tmp_warm_restarts_initial_ti(1_T),
             tmp_warm_restarts_multiplier(2_T),
             tmp_normalization_momentum_average(1_T),
             tmp_normalization_epsilon(1e-1_T);
    
    double tmp_percentage_maximum_thread_usage(25.0);

    enum MyEA::Common::ENUM_TYPE_NETWORKS const type_neural_network(MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_FEEDFORWARD);
    enum MyEA::Common::ENUM_TYPE_LAYER const tmp_type_fully_connected(MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT);
    enum MyEA::Common::ENUM_TYPE_LAYER const tmp_type_pooling_layer(MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING);
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const tmp_type_activation_function_hidden(MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRLU);
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const tmp_type_activation_function_output(MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SOFTMAX);
    enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const tmp_type_loss_function(MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_CROSS_ENTROPY);
    enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS const tmp_type_accuracy_function(MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY);
    enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS const tmp_type_optimizer_function(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad);
    enum MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS const tmp_type_weights_initializer(MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL);
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const tmp_type_normalization(MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION);
    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const tmp_type_dropout(MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP);
    
    std::string tmp_path_dataset_file("sMNIST_28");
    
    // Validate inputs.
    if(tmp_widening_factor_alphas[0u] <= -static_cast<long long int>(tmp_number_units[0u]))
    {
        PRINT_FORMAT("%s: %s: ERROR: Widening factor, alpha[0] (%lld) can not be less or equal to -%zu. At line %d." NEW_LINE,
                     MyEA::String::Get__Time().c_str(),
                     __FUNCTION__,
                     tmp_widening_factor_alphas[0u],
                     tmp_number_units[0u],
                     __LINE__);

        return;
    }
    else if(tmp_widening_factor_alphas[1u] <= -static_cast<long long int>(tmp_number_units[1u]))
    {
        PRINT_FORMAT("%s: %s: ERROR: Widening factor, alpha[0] (%lld) can not be less or equal to -%zu. At line %d." NEW_LINE,
                     MyEA::String::Get__Time().c_str(),
                     __FUNCTION__,
                     tmp_widening_factor_alphas[1u],
                     tmp_number_units[1u],
                     __LINE__);

        return;
    }
    else if(tmp_residual_block_width < 2_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Residual block depth (%zu) can not be less than 2. At line %d." NEW_LINE,
                     MyEA::String::Get__Time().c_str(),
                     __FUNCTION__,
                     tmp_residual_block_width,
                     __LINE__);

        return;
    }
    else if(tmp_use_bottleneck && tmp_residual_block_width == 2_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not use residual bottleneck with a block depth of 2. At line %d." NEW_LINE,
                     MyEA::String::Get__Time().c_str(),
                     __FUNCTION__,
                     __LINE__);

        return;
    }

    switch(tmp_type_fully_connected)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. Need to be the fully connected layer or one of its variant. At line %d." NEW_LINE,
                         MyEA::String::Get__Time().c_str(),
                         __FUNCTION__,
                         tmp_type_fully_connected,
                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_type_fully_connected].c_str(),
                         __LINE__);
                return;
    }

    switch(tmp_type_pooling_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING: break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. Need to be a pooling layer. At line %d." NEW_LINE,
                         MyEA::String::Get__Time().c_str(),
                         __FUNCTION__,
                         tmp_type_pooling_layer,
                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_type_pooling_layer].c_str(),
                         __LINE__);
            return;
    }
    // |END| Validate inputs. |END|

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Dataset name: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_path_dataset_file.c_str());

#if defined(COMPILE_CUDA)
    PRINT_FORMAT("%s: CUDA: Use CUDA: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_CUDA ? "true" : "false");

    PRINT_FORMAT("%s: CUDA: device: %d." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_cuda_device_index);

    if(tmp_use_CUDA)
    {
    #if defined(COMPILE_WINDOWS)
        SetConsoleTitle("Debug - Neural Network [CUDA]");
    #endif

        CUDA__Set__Device(tmp_cuda_device_index);

        CUDA__Reset();
        
        CUDA__Set__Synchronization_Depth(3_zu);

        CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitDevRuntimeSyncDepth, 3u));

        CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_cuda_device, static_cast<int>(tmp_cuda_device_index)));

        CUDA__Print__Device_Property(tmp_cuda_device, tmp_cuda_device_index);
    }
#endif // COMPILE_CUDA
    
    PRINT_FORMAT("%s: OpenMP: Use OpenMP: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_OpenMP ? "true" : "false");
    
    PRINT_FORMAT("%s: OpenMP: Maximum thread usage: %f%%." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_percentage_maximum_thread_usage);
    
    PRINT_FORMAT("%s: Warm restarts: Use warm restarts: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_warm_restarts ? "true" : "false");
    
    PRINT_FORMAT("%s: Warm restarts: minimum learning rate: %f." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 Cast_T(tmp_warm_restarts_minimum_learning_rate));

    PRINT_FORMAT("%s: Warm restarts: maximum learning rate: %f." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 Cast_T(tmp_warm_restarts_maximum_learning_rate));

    PRINT_FORMAT("%s: Warm restarts: Initial Ti: %f." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 Cast_T(tmp_warm_restarts_initial_ti));

    PRINT_FORMAT("%s: Warm restarts: multiplier: %f." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 Cast_T(tmp_warm_restarts_multiplier));

    PRINT_FORMAT("%s: Clip gradient: Use clip gradient: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_clip_gradient ? "true" : "false");
    
    PRINT_FORMAT("%s: Clip gradient: %f." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 Cast_T(tmp_clip_gradient_value));
    
    PRINT_FORMAT("%s: Tied parameter: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_tied_parameter ? "true" : "false");
    
    PRINT_FORMAT("%s: Use bidirectional: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_bidirectional ? "true" : "false");
    
    PRINT_FORMAT("%s: Use residual: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_residual ? "true" : "false");
    
    PRINT_FORMAT("%s: Use pooling: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_pooling ? "true" : "false");
    
    PRINT_FORMAT("%s: Use bottleneck: %s." NEW_LINE,
                 MyEA::String::Get__Time().c_str(),
                 tmp_use_bottleneck ? "true" : "false");
    
    PRINT_FORMAT("%s: Use simulate online training: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_simulate_online_training ? "true" : "false");
    
    PRINT_FORMAT("%s: Use load parameters: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_load_parameters ? "true" : "false");
    
    PRINT_FORMAT("%s: Use dropout: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_dropout ? "true" : "false");
    
    PRINT_FORMAT("%s: Normalization: Use normalization: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_normalization ? "true" : "false");
    
    PRINT_FORMAT("%s: Normalization: Use batch normalization before activation: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_layer_normalization_before_activation ? "true" : "false");
    
    PRINT_FORMAT("%s: Normalization: momentum average: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_normalization_momentum_average));
    
    PRINT_FORMAT("%s: Normalization: epsilon: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_normalization_epsilon));
    
    PRINT_FORMAT("%s: Use save: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_save ? "true" : "false");
    
    PRINT_FORMAT("%s: Use copy: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_copy ? "true" : "false");
    
    PRINT_FORMAT("%s: Use training: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_training ? "true" : "false");
    
    PRINT_FORMAT("%s: Use update BN: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_update_bn ? "true" : "false");
    
    PRINT_FORMAT("%s: Use validating: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_validating ? "true" : "false");
    
    PRINT_FORMAT("%s: Use testing: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_testing ? "true" : "false");
    
    PRINT_FORMAT("%s: Use testing print input: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_testing_print_input ? "true" : "false");
    
    PRINT_FORMAT("%s: Use testing print output: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_use_testing_print_output ? "true" : "false");
    
    PRINT_FORMAT("%s: Type network: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[type_neural_network].c_str());
    
    PRINT_FORMAT("%s: Number layer(s): %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_number_layers);
    
    PRINT_FORMAT("%s: Number unit(s)[0]: %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_number_units[0u]);
    
    PRINT_FORMAT("%s: Number unit(s)[1]: %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_number_units[1u]);
    
    PRINT_FORMAT("%s: Residual block width: %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_residual_block_width);
    
    PRINT_FORMAT("%s: Widening factor, alpha[0]: %lld." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_widening_factor_alphas[0u]);
    
    PRINT_FORMAT("%s: Widening factor, alpha[1]: %lld." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_widening_factor_alphas[1u]);
    
    PRINT_FORMAT("%s: Fully connected layer type: %u | %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_type_fully_connected,
                            MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_type_fully_connected].c_str());
    
    PRINT_FORMAT("%s: Pooling layer type: %u | %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_type_pooling_layer,
                            MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_type_pooling_layer].c_str());
    
    PRINT_FORMAT("%s: Type hidden activation function: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[tmp_type_activation_function_hidden].c_str());

    PRINT_FORMAT("%s: Type output activation function: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[tmp_type_activation_function_output].c_str());

    PRINT_FORMAT("%s: Type loss function: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS_NAMES[tmp_type_loss_function].c_str());

    PRINT_FORMAT("%s: Type accuracy function: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS_NAMES[tmp_type_accuracy_function].c_str());
    
    PRINT_FORMAT("%s: Type optimizer function: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[tmp_type_optimizer_function].c_str());
    
    PRINT_FORMAT("%s: Time delay(s): %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_time_delays);
    
    PRINT_FORMAT("%s: Epoch(s): %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_epochs);
    
    PRINT_FORMAT("%s: Sub-epoch(s): %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_sub_epochs);
    
    PRINT_FORMAT("%s: Length run: %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_length_run);
    
    PRINT_FORMAT("%s: Image size: %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_image_size);
    
    PRINT_FORMAT("%s: Memory allocate: %zu byte(s)." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            tmp_memory_allocate);
    
    PRINT_FORMAT("%s: Regularization L1: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_regularization_l1));
    
    PRINT_FORMAT("%s: Regularization L2: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_regularization_l2));
    
    PRINT_FORMAT("%s: Regularization max-norm constraints: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_regularization_max_norm_constraints));
    
    PRINT_FORMAT("%s: Regularization weight decay: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_regularization_weight_decay));
    
    PRINT_FORMAT("%s: Accuracy variance: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_accuracy_variance));

    PRINT_FORMAT("%s: Activation stepness: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_activation_steepness));
    
    PRINT_FORMAT("%s: Learning rate: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_learning_rate));
    
    PRINT_FORMAT("%s: Learning rate, final: %f." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            Cast_T(tmp_learning_rate_final));
    
    size_t tmp_index_run,
              tmp_index,
              tmp_sub_index;

    double tmp_compute_time,
              tmp_time_total(0.0),
              tmp_widening_factors[2u] = {0},
              tmp_widening_factor_units[2u] = {0};

    T_ *tmp_ptr_array_initial_weights(nullptr),
         tmp_past_error(0_T);

    class Neural_Network *tmp_ptr_neural_network(nullptr);

    struct Layer_Parameters tmp_Layer_Parameters,
                                         tmp_Layer_Parameters_widening;

    std::vector<struct Layer_Parameters> tmp_vector_Layer_Parameters;
    
#if defined(COMPILE_WINDOWS)
    std::chrono::steady_clock::time_point tmp_time_start,
                                                            tmp_time_end;
#elif defined(COMPILE_LINUX)
    std::chrono::_V2::system_clock::time_point tmp_time_start,
                                                                    tmp_time_end;
#endif // COMPILE_WINDOWS || COMPILE_LINUX

    enum MyEA::Common::ENUM_TYPE_DATASET_FILE tmp_type_dataset_file;

    std::string const tmp_path_dimension_parameters_neural_network(tmp_path_dataset_file + ".net"),
                            tmp_path_general_parameters_neural_network(tmp_path_dataset_file + ".nn");

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Dataset: Initialization." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(Input_Dataset_File(tmp_type_dataset_file, tmp_path_dataset_file) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Input_Dataset_File()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }

    class Dataset_Manager<T_> *tmp_ptr_Dataset_Manager(new class Dataset_Manager<T_>(tmp_type_dataset_file, tmp_path_dataset_file));

    if(tmp_ptr_Dataset_Manager == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_Dataset_Manager\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }
    
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_storage = 0;

    tmp_Dataset_Manager_Parameters.type_training = 0;

    if(tmp_ptr_Dataset_Manager->Preparing_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preparing_Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        SAFE_DELETE(tmp_ptr_Dataset_Manager);

        return;
    }

#if defined(COMPILE_CUDA)
    if(tmp_use_CUDA)
    {
        // Memory allocate.
        size_t tmp_memory_total(0),
                  tmp_memory_free(0);

        CUDA__Safe_Call(cudaMemGetInfo(&tmp_memory_free, &tmp_memory_total));

        // Convert bytes to megabytes.
        tmp_memory_free /= KILOBYTE * KILOBYTE;
        // Convert bytes to megabytes.
        tmp_memory_total /= KILOBYTE * KILOBYTE;

        PRINT_FORMAT("%s: GPU: Memory available: %zuMB(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_memory_free);
        PRINT_FORMAT("%s: GPU: Memory used: %zuMB(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_memory_total - tmp_memory_free);
        PRINT_FORMAT("%s: GPU: Memory total: %zuMB(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(), 
                                 tmp_memory_total);

        if(tmp_memory_free < tmp_memory_allocate)
        { tmp_memory_allocate_device = tmp_memory_free; }
        else
        { tmp_memory_allocate_device = tmp_memory_allocate; }

        PRINT_FORMAT("%s: GPU: Memory allocate: %zuMB(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_memory_allocate_device);

        CUDA__Initialize__Device(tmp_cuda_device, tmp_memory_allocate_device * KILOBYTE * KILOBYTE);
        // |END| Memory allocate. |END|

        if(tmp_use_OpenMP)
        {
            PRINT_FORMAT("%s: %s: WARNING: Cannot use OpenMP while using CUDA." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__);

            tmp_use_OpenMP = false;
        }

        if(tmp_ptr_Dataset_Manager->Initialize__CUDA() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__CUDA()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            SAFE_DELETE(tmp_ptr_Dataset_Manager);

            return;
        }
    }
#endif // COMPILE_CUDA

    if(tmp_simulate_online_training && tmp_ptr_Dataset_Manager->Set__Maximum_Data(tmp_ptr_Dataset_Manager->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Data(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_ptr_Dataset_Manager->Get__Number_Examples(),
                                 __LINE__);

        SAFE_DELETE(tmp_ptr_Dataset_Manager);

        return;
    }

    for(tmp_index_run = 0_zu; tmp_index_run != tmp_length_run; ++tmp_index_run)
    {
        tmp_vector_Layer_Parameters.clear();

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Run #%zu" NEW_LINE,
                                MyEA::String::Get__Time().c_str(),
                                tmp_index_run);
        
        if((tmp_ptr_neural_network = new class Neural_Network) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class Neural_Network),
                                     __LINE__);

            break;
        }

        if(tmp_load_parameters)
        {
            PRINT_FORMAT("%s: Neural network: Loading." NEW_LINE, MyEA::String::Get__Time().c_str());

            if(MyEA::File::Path_Exist(tmp_path_dimension_parameters_neural_network) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_path_dimension_parameters_neural_network.c_str(),
                                         __LINE__);

                break;
            }
        
            if(MyEA::File::Path_Exist(tmp_path_general_parameters_neural_network) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_path_general_parameters_neural_network.c_str(),
                                         __LINE__);
                
                break;
            }
            
            PRINT_FORMAT("%s: Neural network: Load from %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_path_dimension_parameters_neural_network.c_str());

            if(tmp_ptr_neural_network->Load(tmp_path_dimension_parameters_neural_network,
                                                            tmp_path_general_parameters_neural_network,
                                                            tmp_memory_allocate) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load(%s, %s, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_path_dimension_parameters_neural_network.c_str(),
                                         tmp_path_general_parameters_neural_network.c_str(),
                                         tmp_memory_allocate,
                                         __LINE__);

                PRINT_FORMAT("%s: Dataset: Deallocate." NEW_LINE, MyEA::String::Get__Time().c_str());
                SAFE_DELETE(tmp_ptr_Dataset_Manager);

                SAFE_DELETE(tmp_ptr_array_initial_weights);
                
                break;
            }
        }
        else
        {
            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
            tmp_Layer_Parameters.unit_parameters[0u]  = tmp_ptr_Dataset_Manager->Get__Number_Inputs();
            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);
            
            PRINT_FORMAT("%s: Neural network: Compile." NEW_LINE, MyEA::String::Get__Time().c_str());
            switch(type_neural_network)
            {
                case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
                    for(tmp_index = 1_zu; tmp_index != tmp_number_layers - 1_zu; ++tmp_index)
                    {
                        tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
                        tmp_Layer_Parameters.unit_parameters[0u] = tmp_number_units[0u];
                        tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);
                        
                        PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                tmp_index,
                                                MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());

                        PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                tmp_index,
                                                tmp_Layer_Parameters.unit_parameters[0u]);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_FEEDFORWARD:
                    if(tmp_use_residual)
                    {
                        if(tmp_number_layers < 6_zu)
                        {
                            
                            PRINT_FORMAT("%s: %s: ERROR: The number of layer(s) (%zu) can not be less than 6. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_number_layers,
                                                     __LINE__);
                            
                            SAFE_DELETE(tmp_ptr_neural_network);
                            
                            break;
                        }
                        
                        size_t const tmp_number_residual_layers((tmp_number_layers - 3_zu) / 3_zu),
                                           tmp_number_residual_layers_last_group((tmp_number_layers - 3_zu) - 2_zu * tmp_number_residual_layers);
                        size_t tmp_residual_unit_index;

                        struct Neural_Network_Initializer tmp_Neural_Network_Initializer;
                        
                        tmp_widening_factors[0u] = static_cast<double>(tmp_widening_factor_alphas[0u]) / static_cast<double>(2_zu * tmp_number_residual_layers + tmp_number_residual_layers_last_group);

                        // First hidden layer.
                        tmp_Layer_Parameters.type_layer = tmp_type_fully_connected;
                        tmp_Layer_Parameters.unit_parameters[0u] = tmp_number_units[0u];
                        tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                        PRINT_FORMAT("%s: Layer[1]: Type: %s." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                        PRINT_FORMAT("%s: Layer[1]: Number neuron unit(s): %zu." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                tmp_Layer_Parameters.unit_parameters[0u]);
                        // |END| First hidden layer. |END|

                        // Residual group #1.
                        tmp_widening_factor_units[0u] = static_cast<double>(tmp_number_units[0u]);

                        for(tmp_residual_unit_index = 0_zu; tmp_residual_unit_index != tmp_number_residual_layers; ++tmp_residual_unit_index)
                        {
                            // Residual unit.
                            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL;
                            tmp_Layer_Parameters.unit_parameters[0u]  = tmp_residual_block_width;

                            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    tmp_vector_Layer_Parameters.size() - 1_zu,
                                                    MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                            PRINT_FORMAT("%s: Layer[%zu]: Block depth: %zu." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    tmp_vector_Layer_Parameters.size() - 1_zu,
                                                    tmp_Layer_Parameters.unit_parameters[0u]);
                            // |END| Residual unit. |END|
                            
                            // Building block.
                            tmp_sub_index = 0_zu;
                            tmp_Layer_Parameters.type_layer = tmp_type_fully_connected;
                            
                            if(tmp_use_bottleneck)
                            {
                                // First hidden layer inside the residual block.
                                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);

                                ++tmp_sub_index;
                                // |END| First hidden layer inside the residual block. |END|

                                // Second hidden layer inside the residual block.
                                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(MyEA::Math::Maximum<double>(tmp_widening_factor_units[0u], tmp_widening_factor_units[0u] + tmp_widening_factors[0u]) / 2.0);

                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);

                                ++tmp_sub_index;
                                // |END| Second hidden layer inside the residual block. |END|
                            }

                            tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                            tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                            for(; tmp_sub_index != tmp_residual_block_width; ++tmp_sub_index)
                            {
                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);
                            }
                            // |END| Building block. |END|
                        }
                        // |END| Residual group #1. |END|

                        // Residual group #2.
                        //  Pooling layer.
                        if(tmp_use_pooling)
                        {
                            tmp_Layer_Parameters.type_layer = tmp_type_pooling_layer;
                            tmp_Layer_Parameters.unit_parameters[0u] = tmp_pooling_kernel_size;
                            tmp_Layer_Parameters.unit_parameters[1u] = tmp_pooling_stride;
                            tmp_Layer_Parameters.unit_parameters[2u] = tmp_pooling_padding;
                            tmp_Layer_Parameters.unit_parameters[3u] = tmp_pooling_dilation;
                            tmp_Layer_Parameters.unit_parameters[4u] = static_cast<size_t>(tmp_pooling_ceil_mode);
                            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                            PRINT_FORMAT("%s: Layer[%zu]: Kernel size: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[0u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Stride: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[1u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Padding: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[2u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Dilation: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[3u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Ceil mode: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[4u] != 0_zu ? "true" : "false");
                        }
                        //  |END| Pooling layer. |END|
                        
                        for(tmp_residual_unit_index = 0_zu; tmp_residual_unit_index != tmp_number_residual_layers; ++tmp_residual_unit_index)
                        {
                            // Residual unit.
                            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL;
                            tmp_Layer_Parameters.unit_parameters[0u]  = tmp_residual_block_width;

                            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                            PRINT_FORMAT("%s: Layer[%zu]: Block depth: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[0u]);
                            // |END| Residual unit. |END|
                            
                            // Building block.
                            tmp_sub_index = 0_zu;
                            tmp_Layer_Parameters.type_layer = tmp_type_fully_connected;
                            
                            if(tmp_use_bottleneck)
                            {
                                // First hidden layer inside the residual block.
                                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);

                                ++tmp_sub_index;
                                // |END| First hidden layer inside the residual block. |END|

                                // Second hidden layer inside the residual block.
                                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(MyEA::Math::Maximum<double>(tmp_widening_factor_units[0u], tmp_widening_factor_units[0u] + tmp_widening_factors[0u]) / 2.0);

                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);

                                ++tmp_sub_index;
                                // |END| Second hidden layer inside the residual block. |END|
                            }

                            tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                            tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                            for(; tmp_sub_index != tmp_residual_block_width; ++tmp_sub_index)
                            {
                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);
                            }
                            // |END| Building block. |END|
                        }
                        // |END| Residual group #2. |END|

                        // Residual group #3.
                        //  Pooling layer.
                        if(tmp_use_pooling)
                        {
                            tmp_Layer_Parameters.type_layer = tmp_type_pooling_layer;
                            tmp_Layer_Parameters.unit_parameters[0u] = tmp_pooling_kernel_size;
                            tmp_Layer_Parameters.unit_parameters[1u] = tmp_pooling_stride;
                            tmp_Layer_Parameters.unit_parameters[2u] = tmp_pooling_padding;
                            tmp_Layer_Parameters.unit_parameters[3u] = tmp_pooling_dilation;
                            tmp_Layer_Parameters.unit_parameters[4u] = static_cast<size_t>(tmp_pooling_ceil_mode);
                            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                            PRINT_FORMAT("%s: Layer[%zu]: Kernel size: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[0u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Stride: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[1u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Padding: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[2u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Dilation: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[3u]);
                            PRINT_FORMAT("%s: Layer[%zu]: Ceil mode: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[4u] != 0_zu ? "true" : "false");
                        }
                        //  |END| Pooling layer. |END|
                        
                        for(tmp_residual_unit_index = 0_zu; tmp_residual_unit_index != tmp_number_residual_layers_last_group; ++tmp_residual_unit_index)
                        {
                            // Residual unit.
                            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL;
                            tmp_Layer_Parameters.unit_parameters[0u]  = tmp_residual_block_width;

                            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                            PRINT_FORMAT("%s: Layer[%zu]: Block depth: %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_vector_Layer_Parameters.size() - 1_zu,
                                                     tmp_Layer_Parameters.unit_parameters[0u]);
                            // |END| Residual unit. |END|
                            
                            // Building block.
                            tmp_sub_index = 0_zu;
                            tmp_Layer_Parameters.type_layer = tmp_type_fully_connected;
                            
                            if(tmp_use_bottleneck)
                            {
                                // First hidden layer inside the residual block.
                                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);

                                ++tmp_sub_index;
                                // |END| First hidden layer inside the residual block. |END|

                                // Second hidden layer inside the residual block.
                                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(MyEA::Math::Maximum<double>(tmp_widening_factor_units[0u], tmp_widening_factor_units[0u] + tmp_widening_factors[0u]) / 2.0);

                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);

                                ++tmp_sub_index;
                                // |END| Second hidden layer inside the residual block. |END|
                            }

                            tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                            tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                            for(; tmp_sub_index != tmp_residual_block_width; ++tmp_sub_index)
                            {
                                tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                                PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                                PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                                         tmp_Layer_Parameters.unit_parameters[0u]);
                            }
                            // |END| Building block. |END|
                        }
                        // |END| Residual group #3. |END|

                        // Last hidden layer.
                        tmp_Layer_Parameters.type_layer = tmp_type_fully_connected;
                        tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);
                        tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);

                        PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 tmp_vector_Layer_Parameters.size() - 1_zu,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                        PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 tmp_vector_Layer_Parameters.size() - 1_zu,
                                                 tmp_Layer_Parameters.unit_parameters[0u]);
                        // |END| Last hidden layer. |END|
                    }
                    else
                    {
                        for(tmp_index = 1_zu; tmp_index != tmp_number_layers - 1_zu; ++tmp_index)
                        {
                            tmp_Layer_Parameters.type_layer = tmp_type_fully_connected;
                            tmp_Layer_Parameters.unit_parameters[0u] = tmp_number_units[0u];
                            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);
                            
                            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_index,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                            PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_index,
                                                     tmp_Layer_Parameters.unit_parameters[0u]);
                        }
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_RECURRENT:
                    for(tmp_index = 1_zu; tmp_index != tmp_number_layers - 1_zu; ++tmp_index)
                    {
                        tmp_Layer_Parameters.use_bidirectional = tmp_use_bidirectional;
                        tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM;
                        tmp_Layer_Parameters.unit_parameters[0u] = tmp_number_units[0u];
                        tmp_Layer_Parameters.unit_parameters[1u]  = tmp_number_units[1u];
                        tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);
                        
                        PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 tmp_index,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());

                        PRINT_FORMAT("%s: Layer[%zu]: Number block unit(s): %zu." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 tmp_index,
                                                 tmp_Layer_Parameters.unit_parameters[0u]);

                        PRINT_FORMAT("%s: Layer[%zu]: Number cell unit(s) per block: %zu." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 tmp_index,
                                                 tmp_Layer_Parameters.unit_parameters[1u]);
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Neural network type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             type_neural_network,
                                             MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[type_neural_network].c_str(),
                                             __LINE__);

                    SAFE_DELETE(tmp_ptr_neural_network);
                        break;
            }
            
            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
            tmp_Layer_Parameters.unit_parameters[0u]  = tmp_ptr_Dataset_Manager->Get__Number_Outputs();
            tmp_vector_Layer_Parameters.push_back(tmp_Layer_Parameters);
            
            PRINT_FORMAT("%s: Number output(s): %zu." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Number_Outputs());

            if(tmp_ptr_neural_network->Compile(tmp_vector_Layer_Parameters.size(),
                                                                 tmp_ptr_Dataset_Manager->Get__Number_Recurrent_Depth(),
                                                                 type_neural_network,
                                                                 tmp_vector_Layer_Parameters.data(),
                                                                 tmp_memory_allocate) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Compile()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                SAFE_DELETE(tmp_ptr_neural_network);

                break;
            }
            else if(tmp_ptr_neural_network->Set__Number_Time_Delays(tmp_time_delays) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Time_Delays(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_time_delays,
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }

            if(tmp_tied_parameter)
            {
                for(tmp_index = 1_zu; tmp_index != tmp_vector_Layer_Parameters.size() - 1_zu; ++tmp_index)
                {
                    if(tmp_ptr_neural_network->Set__Tied_Parameter(tmp_index, true) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(%zu, true)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_index,
                                                 __LINE__);

                        SAFE_DELETE(tmp_ptr_neural_network);

                        break;
                    }
                }
            }

            for(tmp_index = 0_zu; tmp_index != tmp_vector_Layer_Parameters.size(); ++tmp_index)
            {
                if(tmp_ptr_neural_network->Set__Layer_Activation_Steepness(tmp_index, tmp_activation_steepness) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Steepness(%zu, %f)\" function. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_index,
                                                Cast_T(tmp_activation_steepness),
                                                __LINE__);
                    
                    SAFE_DELETE(tmp_ptr_neural_network);
                    
                    break;
                }
            }
            
            for(tmp_index = 1_zu; tmp_index != tmp_vector_Layer_Parameters.size() - 1_zu; ++tmp_index)
            {
                if(tmp_ptr_neural_network->Set__Layer_Activation_Function(tmp_index, tmp_type_activation_function_hidden) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%zu, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_index,
                                             tmp_type_activation_function_hidden,
                                             __LINE__);
                    
                    SAFE_DELETE(tmp_ptr_neural_network);
                    
                    break;
                }
            }
            
            if(tmp_ptr_neural_network->Set__Layer_Activation_Function(tmp_vector_Layer_Parameters.size() - 1_zu, tmp_type_activation_function_output) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%zu, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_vector_Layer_Parameters.size() - 1_zu,
                                         tmp_type_activation_function_output,
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }

            tmp_ptr_neural_network->Set__Loss_Function(tmp_type_loss_function);

            tmp_ptr_neural_network->Set__Accuracy_Function(tmp_type_accuracy_function);
            
            tmp_ptr_neural_network->Set__Optimizer_Function(tmp_type_optimizer_function);
            
            switch(tmp_type_optimizer_function)
            {
                case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
                case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
                case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
                case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
                case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
                case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: tmp_ptr_neural_network->adam_learning_rate = tmp_learning_rate; break;
                default: tmp_ptr_neural_network->learning_rate = tmp_learning_rate; break;
            }
            
            tmp_ptr_neural_network->learning_rate_final = tmp_learning_rate_final;

            if(tmp_use_warm_restarts)
            {
                tmp_ptr_neural_network->use_Warm_Restarts = true;
                tmp_ptr_neural_network->warm_restarts_maximum_learning_rate = tmp_warm_restarts_maximum_learning_rate;
                tmp_ptr_neural_network->warm_restarts_minimum_learning_rate = tmp_warm_restarts_minimum_learning_rate;
                tmp_ptr_neural_network->warm_restarts_initial_T_i = tmp_warm_restarts_initial_ti;
                tmp_ptr_neural_network->warm_restarts_multiplier = tmp_warm_restarts_multiplier;
            }

            tmp_ptr_neural_network->Set__Clip_Gradient(tmp_clip_gradient);

            if(tmp_ptr_neural_network->Set__Clip_Gradient(tmp_clip_gradient_value) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Clip_Gradient(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_clip_gradient_value),
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }

            // Regularization L1.
            // TODO: Each layer having a L1 regularization parameter.
            if(tmp_ptr_neural_network->Set__Regularization__L1(tmp_regularization_l1) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L1(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_regularization_l1),
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }
            // |END| Regularization L1. |END|

            // Regularization L2.
            // TODO: Each layer having a L2 regularization parameter.
            if(tmp_ptr_neural_network->Set__Regularization__L2(tmp_regularization_l2) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L1(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_regularization_l2),
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }
            // |END| Regularization L2. |END|
            
            // Regularization max-norm constraints.
            // TODO: Each layer having a max-norm constraints regularization parameter.
            if(tmp_ptr_neural_network->Set__Regularization__Max_Norm_Constraints(tmp_regularization_max_norm_constraints) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Max_Norm_Constraints(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_regularization_max_norm_constraints),
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }
            // |END| Regularization max-norm constraints. |END|

            // Regularization max-norm constraints.
            // TODO: Each layer having a max-norm constraints regularization parameter.
            if(tmp_ptr_neural_network->Set__Regularization__Weight_Decay(tmp_regularization_weight_decay) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Weight_Decay(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_regularization_weight_decay),
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }
            // |END| Regularization max-norm constraints. |END|

            // Accuracy variance.
            tmp_ptr_neural_network->Set__Accurancy_Variance(tmp_accuracy_variance);
            // |END| Accuracy variance. |END|
            
            // Initialize the weights just at the first run, because sometimes the second initialization differ from the first run.
            if(tmp_index_run == 0_zu)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Weight(s) intialization: %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[tmp_type_weights_initializer].c_str());
               
                PRINT_FORMAT("%s: Seed: %u." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_weights_seed);
                tmp_ptr_neural_network->Class_Generator_Real.Reset();
                tmp_ptr_neural_network->Class_Generator_Gaussian.Reset();
                tmp_ptr_neural_network->Class_Generator_Real.Seed(tmp_weights_seed);
                tmp_ptr_neural_network->Class_Generator_Gaussian.Seed(tmp_weights_seed);

                switch(tmp_type_weights_initializer)
                {
                    case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN: tmp_ptr_neural_network->Initialization__Glorot__Gaussian(); break;
                    case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_UNIFORM: tmp_ptr_neural_network->Initialization__Glorot__Uniform(); break;
                    case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_IDENTITY: tmp_ptr_neural_network->Initialization__Identity(); break;
                    case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LSUV: tmp_ptr_neural_network->Initialize__LSUV(); break;
                    case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL: tmp_ptr_neural_network->Initialization__Orthogonal(); break;
                    case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_UNIFORM:
                        PRINT_FORMAT("%s: Range: [%f, %f]." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 Cast_T(tmp_weights_minimum),
                                                 Cast_T(tmp_weights_maximum));
                        tmp_ptr_neural_network->Class_Generator_Real.Range(tmp_weights_minimum, tmp_weights_maximum);

                        tmp_ptr_neural_network->Initialization__Uniform();
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Weights initializer (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_type_weights_initializer,
                                                 MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[tmp_type_weights_initializer].c_str());
                            break;
                }

                if((tmp_ptr_array_initial_weights = new T_[tmp_ptr_neural_network->total_parameters]) == nullptr)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_neural_network->total_parameters * sizeof(T_),
                                             __LINE__);

                    break;
                }
                memcpy(tmp_ptr_array_initial_weights,
                             tmp_ptr_neural_network->ptr_array_parameters,
                             tmp_ptr_neural_network->total_parameters * sizeof(T_));

                if(tmp_print_parameters)
                {
                    for(size_t w = 0_zu; w != tmp_ptr_neural_network->total_parameters; ++w)
                    {
                        //if(tmp_ptr_array_initial_weights[w] < 0_T)
                        {
                            PRINT_FORMAT("W[%zu]: %f" NEW_LINE, w, Cast_T(tmp_ptr_array_initial_weights[w]));
                        }
                    }
                }
            }
            else
            {
                memcpy(tmp_ptr_neural_network->ptr_array_parameters,
                             tmp_ptr_array_initial_weights,
                             tmp_ptr_neural_network->total_parameters * sizeof(T_));
            }
            
            // Dropout.
            if(tmp_use_dropout)
            {
                switch(tmp_type_dropout)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                        for(tmp_index = 0_zu; tmp_index != tmp_vector_Layer_Parameters.size() - 1_zu; ++tmp_index)
                        {
                            if(tmp_ptr_neural_network->ptr_array_layers[tmp_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED
                               ||
                               tmp_ptr_neural_network->ptr_array_layers[tmp_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
                               ||
                               tmp_ptr_neural_network->ptr_array_layers[tmp_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT)
                            {
                                PRINT_FORMAT("%s: Layer[%zu], Dropout: %s(%f)." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_type_dropout].c_str(),
                                                         Cast_T(tmp_dropout_values[0u]));
                                tmp_ptr_neural_network->Set__Dropout(tmp_index,
                                                                                          tmp_type_dropout,
                                                                                          tmp_dropout_values);
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP:
                        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                        for(tmp_index = 1_zu; tmp_index != tmp_vector_Layer_Parameters.size() - 1_zu; ++tmp_index)
                        {
                            if(tmp_ptr_neural_network->ptr_array_layers[tmp_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
                            {
                                PRINT_FORMAT("%s: Layer[%zu], Dropout: %s(%f)." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_type_dropout].c_str(),
                                                         Cast_T(tmp_dropout_values[0u]));
                                tmp_ptr_neural_network->Set__Dropout(tmp_index,
                                                                                          tmp_type_dropout,
                                                                                          tmp_dropout_values);
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
                        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                        for(tmp_index = 1_zu; tmp_index != tmp_vector_Layer_Parameters.size() - 1_zu; ++tmp_index)
                        {
                            if(tmp_ptr_neural_network->ptr_array_layers[tmp_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM
                               ||
                               tmp_ptr_neural_network->ptr_array_layers[tmp_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_GRU)
                            {
                                PRINT_FORMAT("%s: Layer[%zu], Dropout: %s(%f, %f)." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         tmp_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_type_dropout].c_str(),
                                                         Cast_T(tmp_dropout_values[0u]),
                                                         Cast_T(tmp_dropout_values[1u]));
                                tmp_ptr_neural_network->Set__Dropout(tmp_index,
                                                                                          tmp_type_dropout,
                                                                                          tmp_dropout_values);
                            }
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Dropout type (%u | %s) is not managed in the switch. Need to be the fully connected layer or one of its variant. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_type_dropout,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_type_dropout].c_str(),
                                                 __LINE__);
                            break;
                }
            }
            // |END| Dropout. |END|
            
            // Normalization.
            if(tmp_use_normalization)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                for(tmp_index = 1_zu; tmp_index != tmp_vector_Layer_Parameters.size() - 1u; ++tmp_index)
                {
                    PRINT_FORMAT("%s: Layer[%zu], Normalization: %s." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            tmp_index,
                                            MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[tmp_type_normalization].c_str());
                    
                    tmp_ptr_neural_network->Set__Layer_Normalization(tmp_index, tmp_type_normalization);
                    
                    tmp_ptr_neural_network->ptr_array_layers[tmp_index].use_layer_normalization_before_activation = tmp_use_layer_normalization_before_activation;
                }
                
                tmp_ptr_neural_network->Set__Normalization_Momentum_Average(tmp_normalization_momentum_average);
                
                tmp_ptr_neural_network->Set__Normalization_Epsilon(tmp_normalization_epsilon);
            }
            // |END| Normalization. |END|

            tmp_ptr_neural_network->percentage_maximum_thread_usage = tmp_percentage_maximum_thread_usage;

            if(tmp_ptr_neural_network->Set__OpenMP(tmp_use_OpenMP) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__OpenMP(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_use_OpenMP ? "true" : "false",
                                         __LINE__);
                
                SAFE_DELETE(tmp_ptr_neural_network);
                
                break;
            }
        }

    #if defined(COMPILE_CUDA)
        if(tmp_ptr_neural_network->Set__CUDA(tmp_use_CUDA, tmp_memory_allocate_device) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__CUDA(%s, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_use_CUDA ? "true" : "false",
                                     tmp_memory_allocate_device,
                                     __LINE__);
                
            SAFE_DELETE(tmp_ptr_neural_network);
                
            break;
        }

        if(tmp_use_CUDA && tmp_ptr_neural_network->Initialize__CUDA__Thread(tmp_ptr_Dataset_Manager) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__CUDA__Thread(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
            
            SAFE_DELETE(tmp_ptr_neural_network);
            
            break;
        }
    #endif // COMPILE_CUDA
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Total layer(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_layers);
        
        PRINT_FORMAT("%s: Total basic unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_basic_units);
        
        PRINT_FORMAT("%s: Total neuron unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_neuron_units);
        
        PRINT_FORMAT("%s: Total AF unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_AF_units);
        
        PRINT_FORMAT("%s: Total AF Ind recurrent unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_AF_Ind_recurrent_units);
        
        PRINT_FORMAT("%s: Total normalized unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_normalized_units);
        
        PRINT_FORMAT("%s: Total block unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_block_units);
        
        PRINT_FORMAT("%s: Total cell unit(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_cell_units);
        
        PRINT_FORMAT("%s: Total weight(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_weights);
        
        PRINT_FORMAT("%s: Total parameter(s): %zu" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_neural_network->total_parameters);
        
        // Adept
        if(tmp_use_adept)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Adept: Neural network: Testing on %zu example(s) from the testing set." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Examples());

        #if defined(COMPILE_ADEPT)
            T_ tmp_X(3),
                 tmp_FX,
                 tmp_dX,
                 tmp_S,
                 tmp_dS,
                 tmp_Y,
                 tmp_D,
                 tmp_W[3] = {1.3, -5.3, 4.3},
                 tmp_dW[3] = {0.0},
                 tmp_B(0.4),
                 tmp_dB(0),
                 tmp_Z(0.5);

            auto Summation_Const([](T_ const *const X_received, T_ const *const W_received, T_ const *const B_received, T_ *const summation_received) -> void
            {
                T_ const *const tmp_W(W_received);

                *summation_received += *X_received * tmp_W[0];
                *summation_received += *X_received * tmp_W[1];
                *summation_received += *X_received * tmp_W[2];
                *summation_received += *B_received;
            });

            auto Sigmoid_Const([](T_ const *const summation_received) -> T_ { return(1_T / (1_T + exp(-*summation_received))); });
            
            tmp_FX = 0.5;
            
            adept::active_stack()->new_recording();
            
            //tmp_FX = tmp_X;
            
            Summation_Const(&tmp_FX,
                                        tmp_W,
                                        &tmp_B,
                                        &tmp_S);

            tmp_Y = Sigmoid_Const(&tmp_S);

            tmp_D = tmp_Z - tmp_Y;

            tmp_Y.set_gradient(tmp_D.value());

            adept::active_stack()->reverse();
            
            tmp_dS = AF_SIGMOID_derive(1, tmp_Y) * tmp_D;

            tmp_dX = 0;
            tmp_dX += tmp_dS * tmp_W[0];
            tmp_dX += tmp_dS * tmp_W[1];
            tmp_dX += tmp_dS * tmp_W[2];

            tmp_dW[0] += tmp_dS * tmp_FX;
            tmp_dW[1] += tmp_dS * tmp_FX;
            tmp_dW[2] += tmp_dS * tmp_FX;
            tmp_dB += tmp_dS;

            std::cout << "dJ/dx:" << tmp_X.get_gradient() << std::endl;
            std::cout << "dJ/dx:" << tmp_dX.value() << std::endl;
            std::cout << "dJ/dw[0]:" << tmp_W[0].get_gradient() << std::endl;
            std::cout << "dJ/dw[0]:" << tmp_dW[0].value() << std::endl;
            std::cout << "dJ/dw[1]:" << tmp_W[1].get_gradient() << std::endl;
            std::cout << "dJ/dw[1]:" << tmp_dW[1].value() << std::endl;
            std::cout << "dJ/dw[2]:" << tmp_W[2].get_gradient() << std::endl;
            std::cout << "dJ/dw[2]:" << tmp_dW[2].value() << std::endl;
            std::cout << "dJ/db:" << tmp_B.get_gradient() << std::endl;
            std::cout << "dJ/db:" << tmp_dB.value() << std::endl;
            std::cout << "dJ/ds:" << tmp_S.get_gradient() << std::endl;
            std::cout << "dJ/ds:" << tmp_dS.value() << std::endl;
            std::cout << "dJ/dy:" << tmp_Y.get_gradient() << std::endl;
            
            std::cout << "x:" << tmp_X.value() << std::endl;
            std::cout << "w[0]:" << tmp_W[0].value() << std::endl;
            std::cout << "w[1]:" << tmp_W[1].value() << std::endl;
            std::cout << "w[2]:" << tmp_W[2].value() << std::endl;
            std::cout << "b:" << tmp_B.value() << std::endl;
            std::cout << "s:" << tmp_S.value() << std::endl;
            std::cout << "y:" << tmp_Y.value() << std::endl;
            std::cout << "d:" << tmp_D.value() << std::endl;
            std::cout << "z:" << tmp_Z.value() << std::endl;
        #endif // COMPILE_ADEPT

            tmp_time_start = std::chrono::high_resolution_clock::now();
            
            tmp_ptr_Dataset_Manager->Adept__Gradient(tmp_ptr_neural_network);
            
            tmp_time_end = std::chrono::high_resolution_clock::now();
            
            PRINT_FORMAT("%s: Adept: Neural network: Testing loss: %.9f." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

            PRINT_FORMAT("%s: Adept: Neural network: Testing accurancy: %.5f." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    Cast_T(tmp_ptr_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: Adept: Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());

        }
        // |END| Adept. |END|
        
    #if defined(COMPILE_ADEPT)
        adept::active_stack()->new_recording();
        adept::active_stack()->pause_recording();
    #endif
        
        // Testing.
        if(tmp_use_testing)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Testing on %zu example(s) from the testing set." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Examples());

            tmp_time_start = std::chrono::high_resolution_clock::now();
            
            tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_ptr_neural_network);

            tmp_time_end = std::chrono::high_resolution_clock::now();

            PRINT_FORMAT("%s: Neural network: Testing loss: %.9f." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

            PRINT_FORMAT("%s: Neural network: Testing accurancy: %.5f." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    Cast_T(tmp_ptr_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());
            
            if(tmp_use_testing_print_input || tmp_use_testing_print_output)
            {
                size_t const tmp_number_examples(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Examples()),
                                   tmp_number_inputs(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Inputs()),
                                   tmp_number_outputs(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Outputs()),
                                   tmp_number_recurrent_depth(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Recurrent_Depth());
                size_t tmp_example_index,
                          tmp_time_step,
                          tmp_input_index;
                for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
                {
                    if(tmp_use_testing_print_input)
                    {
                        PRINT_FORMAT("Data[%zu], Input size: %zu" NEW_LINE,
                                                 tmp_example_index,
                                                 tmp_number_inputs);
                    }
                    
                    if(tmp_use_testing_print_output)
                    {
                        PRINT_FORMAT("Data[%zu], Output size: %zu" NEW_LINE,
                                                 tmp_example_index,
                                                 tmp_number_outputs);
                    }

                    for(tmp_time_step = 0_zu; tmp_time_step != tmp_number_recurrent_depth; ++tmp_time_step)
                    {
                        if(tmp_use_testing_print_input)
                        {
                            if(tmp_image_size == 0_zu)
                            {
                                for(tmp_input_index = 0_zu; tmp_input_index != tmp_number_inputs; ++tmp_input_index)
                                {
                                    PRINT_FORMAT("%f ", Cast_T(tmp_ptr_Dataset_Manager->Get__Input_At(tmp_example_index)[tmp_time_step * tmp_number_inputs + tmp_input_index]));
                                }
                            }
                            else
                            {
                                for(tmp_input_index = 0_zu; tmp_input_index != tmp_number_inputs; ++tmp_input_index)
                                {
                                    if(tmp_input_index != 0_zu && tmp_input_index % 28_zu == 0_zu) { PRINT_FORMAT(NEW_LINE); }
                                
                                    PRINT_FORMAT("%.0f ", round(Cast_T(tmp_ptr_Dataset_Manager->Get__Input_At(tmp_example_index)[tmp_time_step * tmp_number_inputs + tmp_input_index])));
                                }
                            }

                            PRINT_FORMAT(NEW_LINE);
                        }
                        
                        if(tmp_use_testing_print_output)
                        {
                            for(tmp_input_index = 0_zu; tmp_input_index != tmp_number_outputs; ++tmp_input_index)
                            {
                                PRINT_FORMAT("%f ", Cast_T(tmp_ptr_neural_network->Get__Outputs(tmp_example_index, tmp_time_step)[tmp_input_index]));
                            }

                            PRINT_FORMAT(NEW_LINE);
                        }
                    }

                    PRINT_FORMAT(NEW_LINE);
                }
            }
            
            if(MyEA::Math::Is_NaN<T_>(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING))) { PAUSE_TERMINAL(); }
        }
        // |END| Testing. |END|
        
        // Validating.
        if(tmp_use_validating)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Validating on %zu example(s) from the validating set." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T_>::Get__Number_Examples());

            tmp_time_start = std::chrono::high_resolution_clock::now();
            
            tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, tmp_ptr_neural_network);

            tmp_time_end = std::chrono::high_resolution_clock::now();

            PRINT_FORMAT("%s: Neural network: Validation loss: %.9f." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));

            PRINT_FORMAT("%s: Neural network: Validation accurancy: %.5f." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    Cast_T(tmp_ptr_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));

            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());
            
            if(MyEA::Math::Is_NaN<T_>(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION))) { PAUSE_TERMINAL(); }
        }
        // |END| Validating. |END|

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Neural network: Train on %zu example(s) from the training set for %zu epoch(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Dataset<T_>::Get__Number_Examples(),
                                 tmp_epochs);

        for(tmp_index = 0_zu; tmp_index != tmp_epochs; ++tmp_index)
        {
            // Simulate online training.
            if(tmp_simulate_online_training)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Simulate online training" NEW_LINE, MyEA::String::Get__Time().c_str());

                T_ *tmp_ptr_array_inputs,
                     *tmp_ptr_array_outputs;

                if((tmp_ptr_array_inputs = new T_[tmp_ptr_Dataset_Manager->Get__Number_Inputs()]) == nullptr)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_Dataset_Manager->Get__Number_Inputs() * sizeof(T_),
                                             __LINE__);

                    break;
                }
                else if((tmp_ptr_array_outputs = new T_[tmp_ptr_Dataset_Manager->Get__Number_Outputs()]) == nullptr)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_Dataset_Manager->Get__Number_Outputs() * sizeof(T_),
                                             __LINE__);

                    break;
                }

                for(size_t k(0_zu); k != tmp_ptr_Dataset_Manager->Get__Number_Inputs(); ++k)
                { tmp_ptr_array_inputs[k] = MATH_RAND(0_T, 1_T); }

                for(size_t k(0_zu); k != tmp_ptr_Dataset_Manager->Get__Number_Outputs(); ++k)
                { tmp_ptr_array_outputs[k] = static_cast<T_>(rand() & 1); }

                tmp_ptr_Dataset_Manager->Push_Back(tmp_ptr_array_inputs, tmp_ptr_array_outputs);
                
                /*
                for(unsigned int k(0_zu), o; k != tmp_ptr_Dataset_Manager->Get__Number_Examples(); ++k)
                {
                    for(o = 0_zu; o != tmp_ptr_Dataset_Manager->Get__Number_Inputs(); ++o)
                    { PRINT_FORMAT("Input[%u][%u] = %f" NEW_LINE, k, o, tmp_ptr_Dataset_Manager->Get__Input_At(k)[o]); }

                    for(o = 0_zu; o != tmp_ptr_Dataset_Manager->Get__Number_Outputs(); ++o)
                    { PRINT_FORMAT("Output[%u][%u] = %f" NEW_LINE, k, o, tmp_ptr_Dataset_Manager->Get__Output_At(k)[o]); }
                }
                */

                delete[](tmp_ptr_array_inputs);
                delete[](tmp_ptr_array_outputs);
            }
            // |END| Simulate online training. |END|

            tmp_time_start = std::chrono::high_resolution_clock::now();
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Train for %zu sub-epoch(s)." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_sub_epochs);

            if(tmp_use_training)
            {
                for(size_t tmp_sub_epoch_index(0_zu); tmp_sub_epoch_index != tmp_sub_epochs; ++tmp_sub_epoch_index)
                {
                    tmp_ptr_Dataset_Manager->Training(tmp_ptr_neural_network);
                }
            }

            tmp_time_end = std::chrono::high_resolution_clock::now();

            PRINT_FORMAT("%s: Neural network: Training loss: %.9f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));

            PRINT_FORMAT("%s: Neural network: Training accuracy: %.5f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
            
            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: [%zu] Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_index,
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());

            tmp_time_total += tmp_compute_time;

            if(MyEA::Math::Is_NaN<T_>(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING))) { PAUSE_TERMINAL(); }
        }

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Time elapse total: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::String::Get__Time_Elapse(tmp_time_total).c_str());
        
        // Update BN moving average.
        if(tmp_use_update_bn)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Update batch normalization on %zu example(s) from the training set." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Dataset<T_>::Get__Number_Examples());

            tmp_time_start = std::chrono::high_resolution_clock::now();

            std::pair<T_, T_> const tmp_pair(tmp_ptr_Dataset_Manager->Type_Update_Batch_Normalization(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_ptr_neural_network));
            
            tmp_time_end = std::chrono::high_resolution_clock::now();

            PRINT_FORMAT("%s: Neural network: Testing loss: %.9f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(std::get<0>(tmp_pair)));

            PRINT_FORMAT("%s: Neural network: Testing accurancy: %.5f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(std::get<1>(tmp_pair)));

            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());
        }

        // Testing
        if(tmp_use_testing)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Testing on %zu example(s) from the testing set." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Examples());

            tmp_time_start = std::chrono::high_resolution_clock::now();

            tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_ptr_neural_network);

            tmp_time_end = std::chrono::high_resolution_clock::now();

            PRINT_FORMAT("%s: Neural network: Testing loss: %.9f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

            PRINT_FORMAT("%s: Neural network: Testing accurancy: %.5f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());
            
            if(MyEA::Math::Is_NaN<T_>(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING))) { PAUSE_TERMINAL(); }

            if(tmp_use_testing_print_input || tmp_use_testing_print_output)
            {
                size_t const tmp_number_examples(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Examples()),
                                   tmp_number_inputs(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Inputs()),
                                   tmp_number_outputs(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Outputs()),
                                   tmp_number_recurrent_depth(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Recurrent_Depth());
                size_t tmp_example_index,
                          tmp_time_step,
                          tmp_input_index;
                for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
                {
                    if(tmp_use_testing_print_input)
                    {
                        PRINT_FORMAT("Data[%zu], Input size: %zu" NEW_LINE,
                                                 tmp_example_index,
                                                 tmp_number_inputs);
                    }
                    
                    if(tmp_use_testing_print_output)
                    {
                        PRINT_FORMAT("Data[%zu], Output size: %zu" NEW_LINE,
                                                 tmp_example_index,
                                                 tmp_number_outputs);
                    }

                    for(tmp_time_step = 0_zu; tmp_time_step != tmp_number_recurrent_depth; ++tmp_time_step)
                    {
                        if(tmp_use_testing_print_input)
                        {
                            if(tmp_image_size == 0_zu)
                            {
                                for(tmp_input_index = 0_zu; tmp_input_index != tmp_number_inputs; ++tmp_input_index)
                                {
                                    PRINT_FORMAT("%f ", Cast_T(tmp_ptr_Dataset_Manager->Get__Input_At(tmp_example_index)[tmp_time_step * tmp_number_inputs + tmp_input_index]));
                                }
                            }
                            else
                            {
                                for(tmp_input_index = 0_zu; tmp_input_index != tmp_number_inputs; ++tmp_input_index)
                                {
                                    if(tmp_input_index != 0_zu && tmp_input_index % 28_zu == 0_zu) { PRINT_FORMAT(NEW_LINE); }
                                
                                    PRINT_FORMAT("%.0f ", round(Cast_T(tmp_ptr_Dataset_Manager->Get__Input_At(tmp_example_index)[tmp_time_step * tmp_number_inputs + tmp_input_index])));
                                }
                            }

                            PRINT_FORMAT(NEW_LINE);
                        }
                        
                        if(tmp_use_testing_print_output)
                        {
                            for(tmp_input_index = 0_zu; tmp_input_index != tmp_number_outputs; ++tmp_input_index)
                            {
                                PRINT_FORMAT("%f ", Cast_T(tmp_ptr_neural_network->Get__Outputs(tmp_example_index, tmp_time_step)[tmp_input_index]));
                            }

                            PRINT_FORMAT(NEW_LINE);
                        }
                    }

                    PRINT_FORMAT(NEW_LINE);
                }
            }
        }
        // |END| Testing |END|
        
        // Validating
        if(tmp_use_validating)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Validating on %zu example(s) from the validating set." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T_>::Get__Number_Examples());

            tmp_time_start = std::chrono::high_resolution_clock::now();

            tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, tmp_ptr_neural_network);

            tmp_time_end = std::chrono::high_resolution_clock::now();

            PRINT_FORMAT("%s: Neural network: Validation loss: %.9f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));

            PRINT_FORMAT("%s: Neural network: Validation accurancy: %.5f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));

            tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

            PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());
            
            if(MyEA::Math::Is_NaN<T_>(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION))) { PAUSE_TERMINAL(); }
        }
        // |END| Validating |END|
        
        if(tmp_copy)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Neural network: Copy." NEW_LINE, MyEA::String::Get__Time().c_str());
            
            class Neural_Network *tmp_ptr_trained_neural_network;
            
            if((tmp_ptr_trained_neural_network = new class Neural_Network) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            sizeof(class Neural_Network),
                                            __LINE__);

                break;
            }

            if(tmp_ptr_trained_neural_network->Copy(*tmp_ptr_neural_network) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
            }
            
            // Testing
            if(tmp_use_testing)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Neural network: Testing on %zu example(s) from the testing set." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T_>::Get__Number_Examples());

                tmp_time_start = std::chrono::high_resolution_clock::now();

                tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_ptr_trained_neural_network);

                tmp_time_end = std::chrono::high_resolution_clock::now();

                PRINT_FORMAT("%s: Neural network: Testing loss: %.9f." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_ptr_trained_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

                PRINT_FORMAT("%s: Neural network: Testing accurancy: %.5f." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_ptr_trained_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));

                tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

                PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());

                if(MyEA::Math::Is_NaN<T_>(tmp_ptr_trained_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING))) { PAUSE_TERMINAL(); }
            }
            // |END| Testing |END|
            
            // Validating
            if(tmp_use_validating)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Neural network: Validating on %zu example(s) from the validating set." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T_>::Get__Number_Examples());

                tmp_time_start = std::chrono::high_resolution_clock::now();

                tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, tmp_ptr_trained_neural_network);

                tmp_time_end = std::chrono::high_resolution_clock::now();

                PRINT_FORMAT("%s: Neural network: Validation loss: %.9f." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_ptr_trained_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));

                PRINT_FORMAT("%s: Neural network: Validation accurancy: %.5f." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_ptr_trained_neural_network->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));

                tmp_compute_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count()) / 1e9;

                PRINT_FORMAT("%s: Time elapse: %s" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         MyEA::String::Get__Time_Elapse(tmp_compute_time).c_str());

                if(MyEA::Math::Is_NaN<T_>(tmp_ptr_trained_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION))) { PAUSE_TERMINAL(); }
            }
            // |END| Validating |END|

            if(tmp_save)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Saving neural network dimension parameters to %s." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_path_dimension_parameters_neural_network.c_str());
        
                if(tmp_ptr_neural_network->Save_Dimension_Parameters(tmp_path_dimension_parameters_neural_network) == false)
                {
                    PRINT_FORMAT("%s: ERROR: A error has been return while saving dimension parameters to %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             tmp_path_dimension_parameters_neural_network.c_str());
                }
        
                PRINT_FORMAT("%s: Saving neural network general parameters to %s." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_path_general_parameters_neural_network.c_str());

                if(tmp_ptr_neural_network->Save_General_Parameters(tmp_path_general_parameters_neural_network) == false)
                {
                    PRINT_FORMAT("%s: ERROR: A error has been return while saving general parameters to %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             tmp_path_general_parameters_neural_network.c_str());
                }

                PRINT_FORMAT("%s" NEW_LINE, tmp_ptr_neural_network->Get__Parameters().c_str());
            }

            delete(tmp_ptr_trained_neural_network);
        }
        else if(tmp_save)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Saving neural network dimension parameters to %s." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_path_dimension_parameters_neural_network.c_str());
        
            if(tmp_ptr_neural_network->Save_Dimension_Parameters(tmp_path_dimension_parameters_neural_network) == false)
            {
                PRINT_FORMAT("%s: ERROR: A error has been return while saving dimension parameters to %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_path_dimension_parameters_neural_network.c_str());
            }
        
            PRINT_FORMAT("%s: Saving neural network general parameters to %s." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_path_general_parameters_neural_network.c_str());

            if(tmp_ptr_neural_network->Save_General_Parameters(tmp_path_general_parameters_neural_network) == false)
            {
                PRINT_FORMAT("%s: ERROR: A error has been return while saving general parameters to %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_path_general_parameters_neural_network.c_str());
            }

            PRINT_FORMAT("%s" NEW_LINE, tmp_ptr_neural_network->Get__Parameters().c_str());
        }

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Neural network: Deallocate and delete." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        if(tmp_index_run == 0_zu) { tmp_past_error = tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING); }
        else if(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING) != tmp_past_error)
        {
            PRINT_FORMAT("%s: ERROR: %.9f != %.9f" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(tmp_ptr_neural_network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)),
                                     Cast_T(tmp_past_error));
            
            SAFE_DELETE(tmp_ptr_neural_network);
            
            break;
        }

        PRINT_FORMAT("%s: Neural network: Deallocate." NEW_LINE, MyEA::String::Get__Time().c_str());
        SAFE_DELETE(tmp_ptr_neural_network);
    }
    
    delete[](tmp_ptr_array_initial_weights);

    PRINT_FORMAT("%s: Dataset: Deallocate." NEW_LINE, MyEA::String::Get__Time().c_str());
    SAFE_DELETE(tmp_ptr_Dataset_Manager);
}