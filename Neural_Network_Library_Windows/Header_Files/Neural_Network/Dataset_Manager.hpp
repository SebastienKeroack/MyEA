#pragma once

#include <Capturing/Shutdown/Shutdown.hpp>
#include <Enums/Enum_Type_Dataset.hpp>
#include <Enums/Enum_Type_Dataset_File.hpp>
#include <Enums/Enum_Type_Dataset_Process.hpp>
#include <Enums/Enum_Type_Dataset_Manager_Storage.hpp>
#include <Random/Bernoulli.hpp>
#include <Random/Floating.hpp>
#include <Random/Gaussian.hpp>
#include <Random/Integer.hpp>
#include <Tools/While_Condition.hpp>
#include <Math/Math.hpp>

#include <Neural_Network/Neural_Network.hpp>

#include <atomic>

enum ENUM_TYPE_INPUT : unsigned int
{
    TYPE_INPUT_INPUT = 0u,
    TYPE_INPUT_OUTPUT = 1u,
    TYPE_INPUT_LENGTH = 2u
};

bool Input_Dataset_File(enum MyEA::Common::ENUM_TYPE_DATASET_FILE &ref_type_dateset_file_received, std::string const &ref_path_received);

template<typename T>
bool Append_To_Dataset_File(size_t const size_inputs_received,
                                            size_t const size_outputs_received,
                                            size_t const size_recurrent_depth_received,
                                            T const *const ptr_array_inputs_received,
                                            T const *const ptr_array_outputs_received,
                                            std::string &ref_path_received);

template<typename T>
bool Time_Direction(size_t const number_outputs_received,
                              size_t const number_recurrent_depth_received,
                              T const minimum_range_received,
                              T const maximum_range_received,
                              T *const ptr_array_outputs_received);

template<typename T>
struct Scaler__Minimum_Maximum
{
    Scaler__Minimum_Maximum(void) { }

    struct Scaler__Minimum_Maximum<T>& operator=(struct Scaler__Minimum_Maximum<T> const &ref_Scaler__Minimum_Maximum_received);

    void Copy(struct Scaler__Minimum_Maximum<T> const &ref_Scaler__Minimum_Maximum_received);

    size_t start_index = 0_zu;
    size_t end_index = 0_zu;

    T minimum_value = 0;
    T maximum_value =1;
    T minimum_range = 0;
    T maximum_range = 1;
};

template<typename T>
struct Scaler__Zero_Centered
{
    Scaler__Zero_Centered(void) { }

    struct Scaler__Zero_Centered<T>& operator=(struct Scaler__Zero_Centered<T> const &ref_Scaler__Zero_Centered_received);

    void Copy(struct Scaler__Zero_Centered<T> const &ref_Scaler__Zero_Centered_received);

    size_t start_index = 0_zu;
    size_t end_index = 0_zu;

    T multiplier = 0;
};

template<typename T>
class Dataset
{
    protected:
        size_t p_start_index = 0_zu;

        size_t p_number_examples = 0_zu;
        
        size_t p_number_examples_allocated = 0_zu;
        
        size_t p_number_recurrent_depth = 0_zu;
        
        size_t p_number_inputs = 0_zu;
        
        size_t p_number_outputs = 0_zu;
        
        size_t p_file_buffer_size = 32_zu * KILOBYTE * KILOBYTE; // byte(s).

        size_t p_file_buffer_shift_size = 256_zu * KILOBYTE; // byte(s).
        
        size_t *p_ptr_input_array_coefficient_matrix_size = nullptr;
        
        size_t *p_ptr_output_array_coefficient_matrix_size = nullptr;

        T const **p_ptr_array_inputs_array = nullptr;
        
        T const **p_ptr_array_outputs_array = nullptr;
        
        T *p_ptr_array_inputs = nullptr;
        
        T *p_ptr_array_outputs = nullptr;
        
        T **p_ptr_input_coefficient_matrix = nullptr;
        
        T **p_ptr_output_coefficient_matrix = nullptr;

        enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH;

        enum MyEA::Common::ENUM_TYPE_DATASET_FILE p_type_data_file = MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET_SPLIT;

    public:
        Dataset(void);
        
        Dataset(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received);
        
        virtual ~Dataset(void);

        class Dataset<T>& operator=(class Dataset<T> const &ref_Dataset_received);
        
        void Copy(class Dataset<T> const &ref_Dataset_received);
        
        void Reference(size_t const number_examples_received,
                              T const **ptr_array_inputs_array_received,
                              T const **ptr_array_outputs_array_received,
                              class Dataset<T> const &ref_Dataset_received);
        
        void Train_Epoch_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        
        void Train_Epoch_Loop(class Neural_Network *const ptr_Neural_Network_received);
        
        void Check_Use__Label(void);
        
        void Compute__Start_Index(void);

        void Adept__Gradient(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual void Train_Batch_BP_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual void Train_Batch_BP_Loop(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual bool Initialize(void);
        
        bool Set__Type_Data_File(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_dataset_file_received);

        bool Save(std::string const &ref_path_received, bool const normalize_received = false);
        
        bool Save(class Neural_Network *const ptr_Autoencoder_received, std::string const path_received);
        
        bool Shift_Entries(size_t const shift_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Time_Direction(T const minimum_range_received,
                                      T const maximum_range_received,
                                      enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Input_To_Output(enum ENUM_TYPE_INPUT const type_input_received);

        bool Unrecurrent(void);
        
        bool Remove(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received);

        bool Allocate(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received);
        
        bool Remove_Duplicate(void);
        
        bool Spliting_Dataset(size_t const desired_data_per_file_received, std::string const &ref_path_file_received);
        
        bool Simulate_Classification_Trading_Session(class Neural_Network *const ptr_Neural_Network_received);
        
        bool Replace_Entries(class Dataset<T> const *const ptr_source_Dataset_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Replace_Entries(class Dataset<T> const *const ptr_Autoencoder_Dataset_received, class Neural_Network *const ptr_Autoencoder_received);

        bool Concat(class Dataset<T> const *const ptr_source_Dataset_received);
        
        bool Save__Dataset_Custom(std::string const &ref_path_received); // WARNING

        bool Save__Sequential_Input(size_t const number_recurrent_depth_received, std::string const &ref_path_received);
        
        bool Preprocessing__Minimum_Maximum(size_t const data_start_index_received,
                                                                      size_t const data_end_index_received,
                                                                      T const minimum_value_received,
                                                                      T const maximum_value_received,
                                                                      T const minimum_range_received,
                                                                      T const maximum_range_received,
                                                                      enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum(size_t const data_start_index_received,
                                                                      size_t const data_end_index_received,
                                                                      size_t const input_index_received,
                                                                      T const minimum_value_received,
                                                                      T const maximum_value_received,
                                                                      T const minimum_range_received,
                                                                      T const maximum_range_received,
                                                                      enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum(T *const ptr_array_inputs_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum(size_t const input_index_received,
                                                                      T *const ptr_array_inputs_received,
                                                                      enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum_Inverse(enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum_Inverse(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum_Inverse(T *const ptr_array_inputs_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Minimum_Maximum_Inverse(size_t const input_index_received,
                                                                                 T *const ptr_array_inputs_received,
                                                                                 enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Zero_Centered(size_t const data_start_index_received,
                                                             size_t const data_end_index_received,
                                                             T const multiplier_received,
                                                             enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Zero_Centered(size_t const data_start_index_received,
                                                             size_t const data_end_index_received,
                                                             size_t const input_index_received,
                                                             T const multiplier_received,
                                                             enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Zero_Centered(T *const ptr_array_inputs_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Zero_Centered(size_t const input_index_received,
                                                             T *const ptr_array_inputs_received,
                                                             enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Zero_Centered_Inverse(enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Zero_Centered_Inverse(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__MODWT(size_t const desired_J_level_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__MODWT(size_t const input_index_received,
                                                     size_t const desired_J_level_received,
                                                     enum ENUM_TYPE_INPUT const type_input_received);
        
        /* Preprocess the input(s) array with the past datapoint in the dataset.
           The array should not be present inside the dataset!
           Should be call sequentialy w.r.t dataset order. */
        bool Preprocessing__MODWT(size_t const input_index_received,
                                                     T *const ptr_array_inputs_received,
                                                     enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__MODWT_Inverse(enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__MODWT_Inverse(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Merge__MODWT(size_t const desired_J_level_received, enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Merge__MODWT(size_t const input_index_received,
                                                                 size_t const desired_J_level_received,
                                                                 enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Merge__MODWT(size_t const input_index_received,
                                                                 size_t const input_size_received,
                                                                 T *&ptr_array_inputs_received,
                                                                 enum ENUM_TYPE_INPUT const type_input_received);
        
        bool Preprocessing__Sequence_Window(size_t const sequence_window_received,
                                                                    size_t const sequence_horizon_received,
                                                                    T *&ptr_array_inputs_received);

        bool Check_Topology(size_t const number_inputs_received,
                                        size_t const number_outputs_received,
                                        size_t const number_recurrent_depth_received) const;
        
        // Check if the object is a reference.
        bool Get__Reference(void) const;
        
        bool Use__Multi_Label(void) const;
        
        virtual bool Deallocate(void);

        size_t Get__Identical_Outputs(std::vector<T> const &ref_vector_identical_outputs_received);
        
        virtual size_t Get__Number_Examples(void) const;
        
        virtual size_t Get__Number_Batch(void) const;
        
        size_t Get__Number_Inputs(void) const;
        
        size_t Get__Number_Outputs(void) const;
        
        size_t Get__Number_Recurrent_Depth(void) const;

        size_t MODWT__J_Level_Maximum(void) const;

        T Measure_Accuracy(size_t const batch_size_received,
                                        T const *const *const ptr_array_inputs_received,
                                        T const *const *const ptr_array_desired_outputs_received,
                                        class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Training(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Training_Loop(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Testing(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Testing_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Testing_Loop(class Neural_Network *const ptr_Neural_Network_received);

        enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS Get__Type_Dataset_Process(void) const;
        
        T Get__Minimum_Input(size_t const data_start_index_received,
                                           size_t const data_end_index_received,
                                           size_t const input_index_received,
                                           enum ENUM_TYPE_INPUT const type_input_received) const;
        
        T Get__Minimum_Input(size_t const data_start_index_received,
                                           size_t const data_end_index_received,
                                           enum ENUM_TYPE_INPUT const type_input_received) const;
        
        T Get__Maximum_Input(size_t const data_start_index_received,
                                           size_t const data_end_index_received,
                                           size_t const input_index_received,
                                           enum ENUM_TYPE_INPUT const type_input_received) const;
        
        T Get__Maximum_Input(size_t const data_start_index_received,
                                           size_t const data_end_index_received,
                                           enum ENUM_TYPE_INPUT const type_input_received) const;
        
        virtual T Get__Input_At(size_t const index_received, size_t const sub_index_received) const;
        
        virtual T Get__Output_At(size_t const index_received, size_t const sub_index_received) const;
        
        virtual T const *const Get__Input_At(size_t const index_received) const;
        
        virtual T const *const Get__Output_At(size_t const index_received) const;
        
        virtual T const *const *const Get__Input_Array(void) const;
        
        virtual T const *const *const Get__Output_Array(void) const;

        size_t Get__Sizeof(void);

        struct Scaler__Minimum_Maximum<T> *const Get__Scalar__Minimum_Maximum(enum ENUM_TYPE_INPUT const type_input_received) const;
        
        struct Scaler__Zero_Centered<T> *const Get__Scalar__Zero_Centered(enum ENUM_TYPE_INPUT const type_input_received) const;

    private:
        bool Shift_Arrays(size_t const input_index_received,
                                  size_t const shift_size_received,
                                  enum ENUM_TYPE_INPUT const type_input_received);

        bool Save__Dataset(std::string const &ref_path_received, bool const normalize_received = false);
        
        bool Save__Dataset_Split(std::string const &ref_path_received);
        
        bool Save__Dataset_Split__Input(std::string const &ref_path_received);
        
        bool Save__Dataset_Split__Output(std::string const &ref_path_received);
        
        bool Allocate__Dataset(std::string const &ref_path_received);
        
        bool Allocate__Dataset_Split(std::string const &ref_path_received);
        
        bool Allocate__Dataset_Split__Input(std::string const &ref_path_received);
        
        bool Allocate__Dataset_Split__Output(std::string const &ref_path_received);
        
        bool Allocate__MNIST(std::string const &ref_path_received);
        
        bool _reference = false;
        
        bool _use_multi_label = false;

        struct Scaler__Minimum_Maximum<T> *_ptr_input_array_scaler__minimum_maximum = nullptr;
        
        struct Scaler__Minimum_Maximum<T> *_ptr_output_array_scaler__minimum_maximum = nullptr;

        struct Scaler__Zero_Centered<T> *_ptr_input_array_scaler__zero_centered = nullptr;
        
        struct Scaler__Zero_Centered<T> *_ptr_output_array_scaler__zero_centered = nullptr;
};

template<typename T>
class Dataset_Mini_Batch : public Dataset<T>
{
    public:
        Dataset_Mini_Batch(void);
        Dataset_Mini_Batch(bool const use_shuffle_received,
                                      size_t const desired_number_examples_per_mini_batch_received,
                                      size_t const number_mini_batch_maximum_received,
                                      class Dataset<T> &ref_Dataset_received);
        virtual ~Dataset_Mini_Batch(void);

        void Shuffle(void);
        void Set__Use__Shuffle(bool const use_shuffle_received);
        void Reset(void);

        virtual bool Initialize(void);
        bool Initialize(bool const use_shuffle_received,
                            size_t const desired_number_examples_per_mini_batch_received,
                            size_t const number_mini_batch_maximum_received);
        bool Set__Desired_Data_Per_Batch(size_t const desired_number_examples_per_mini_batch_received, size_t const number_mini_batch_maximum_received = 0_zu);
        bool Increment_Mini_Batch(size_t const mini_batch_iteration_received);
        bool Get__Use__Shuffle(void) const;
        virtual bool Deallocate(void);
        bool use_shuffle = true;

        virtual size_t Get__Number_Examples(void) const;
        virtual size_t Get__Number_Batch(void) const;
        size_t Get__Number_Examples_Per_Batch(void) const;
        size_t Get__Number_Examples_Last_Batch(void) const;
        size_t number_examples = 0_zu;
        size_t number_mini_batch = 0_zu;
        size_t number_examples_per_iteration = 0_zu;
        size_t number_examples_last_iteration = 0_zu;
        size_t *ptr_array_stochastic_index = nullptr;

        virtual T Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        virtual T Training_Loop(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Get__Input_At(size_t const index_received, size_t const sub_index_received) const;
        virtual T Get__Output_At(size_t const index_received, size_t const sub_index_received) const;
        virtual T const *const Get__Input_At(size_t const index_received) const;
        virtual T const *const Get__Output_At(size_t const index_received) const;
        virtual T const *const *const Get__Input_Array(void) const;
        virtual T const *const *const Get__Output_Array(void) const;
        T const **ptr_array_inputs_array_stochastic = nullptr;
        T const **ptr_array_outputs_array_stochastic = nullptr;

        class MyEA::Random::Integer<size_t> Generator_Random;
};

template<typename T>
class Dataset_Cross_Validation : public Dataset<T>
{
    protected:
        T Test_Epoch_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        T Test_Epoch_Loop(class Neural_Network *const ptr_Neural_Network_received);
    
    public:
        Dataset_Cross_Validation(void);
        virtual ~Dataset_Cross_Validation(void);

        void Shuffle(void);
        void Set__Use__Shuffle(bool const use_shuffle_received);
        void Reset(void);

        virtual bool Initialize(void);
        /* number_k_sub_fold:
            = 0: number_k_fold - 1.
            = 1: no mini batch fold.
            > 1: mini batch fold. */
        bool Initialize__Fold(bool const use_shuffle_received,
                                      size_t const number_k_fold_received,
                                      size_t const number_k_sub_fold_received);
        bool Set__Desired_K_Fold(size_t const number_k_fold_received, size_t const number_k_sub_fold_received);
        bool Increment_Fold(size_t const fold_received);
        bool Increment_Sub_Fold(size_t const sub_fold_received);
        bool Get__Use__Shuffle(void) const;
        virtual bool Deallocate(void);
        bool use_shuffle = true;


        virtual size_t Get__Number_Examples(void) const;
        virtual size_t Get__Number_Batch(void) const;
        size_t Get__Number_Sub_Batch(void) const;
        size_t Get__Number_Examples_Training(void) const;
        size_t Get__Number_Examples_Validating(void) const;
        size_t Get__Number_Examples_Per_Fold(void) const;
        size_t Get__Number_Examples_Per_Sub_Iteration(void) const;
        size_t Get__Number_Examples_Last_Sub_Iteration(void) const;
        size_t number_examples = 0_zu;
        size_t number_k_fold = 0_zu;
        size_t number_k_sub_fold = 0_zu;
        size_t number_examples_per_fold = 0_zu;
        size_t number_examples_training = 0_zu;
        size_t number_examples_validating = 0_zu;
        size_t number_examples_per_sub_iteration = 0_zu;
        size_t number_examples_last_sub_iteration = 0_zu;
        size_t *ptr_array_stochastic_index = nullptr;

        virtual T Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        virtual T Training_Loop(class Neural_Network *const ptr_Neural_Network_received);
        
        virtual T Get__Input_At(size_t const index_received, size_t const sub_index_received) const;
        virtual T Get__Output_At(size_t const index_received, size_t const sub_index_received) const;
        virtual T const *const Get__Input_At(size_t const index_received) const;
        virtual T const *const Get__Output_At(size_t const index_received) const;
        virtual T const *const *const Get__Input_Array(void) const;
        virtual T const *const *const Get__Output_Array(void) const;
        T const **ptr_array_inputs_array_k_fold = nullptr;
        T const **ptr_array_outputs_array_k_fold = nullptr;
        T const **ptr_array_inputs_array_k_sub_fold = nullptr;
        T const **ptr_array_outputs_array_k_sub_fold = nullptr;
        T const **ptr_array_inputs_array_validation = nullptr;
        T const **ptr_array_outputs_array_validation = nullptr;

        class MyEA::Random::Integer<size_t> Generator_Random;
};

template<typename T>
class Gaussian_Search
{
    protected:
        class Dataset_Manager<T> **p_ptr_array_ptr_dataset_manager = nullptr;

        class Neural_Network **p_ptr_array_ptr_individuals = nullptr;

    public:
        Gaussian_Search(void);
        ~Gaussian_Search(void);

        void Deallocate__Dataset_Manager(void);
        void Deallocate__Population(void);
        void Deallocate(void);
        
        bool Initialize__OpenMP(void);
        bool Set__OpenMP(bool const use_openmp_received);
        bool Set__Population_Size(size_t const population_size_received);
        bool Set__Population_Gaussian(double const population_gaussian_percent_received);
        bool Set__Maximum_Thread_Usage(double const percentage_maximum_thread_usage_received);
        bool Allouable__Thread_Size(size_t const desired_number_threads_received, size_t &ref_number_threads_allouable_received);
        bool Update__Thread_Size(size_t const desired_number_threads_received);
        bool Update__Thread_Size__Population(size_t const desired_number_threads_received);
        bool Update__Batch_Size__Population(size_t const desired_batch_size_received);
        bool Update__Population(class Neural_Network *const ptr_source_Neural_Network_received);
        bool Update__Dataset_Manager(class Dataset_Manager<T> *const ptr_source_Dataset_Manager_received);
        bool Optimize(size_t const number_iterations_received,
                             class Dataset_Manager<T> *const ptr_Dataset_Manager_received,
                             class Neural_Network *const ptr_Neural_Network_received);
        bool Evaluation(void);
        bool Evaluation(class Dataset_Manager<T> *const ptr_Dataset_Manager_received);
        bool User_Controls(void);
        bool User_Controls__Push_Back(void);
        bool User_Controls__Hyperparameter_Manager(void);
        bool User_Controls__OpenMP(void);
        // Index: Layer/Unit index.
        bool Push_Back(int const hyper_parameter_id_received,
                                 size_t const index_received,
                                 T const value_received,
                                 T const minimum_value_received,
                                 T const maximum_value_received,
                                 T const variance_received = T(0.1));
        bool Initialize__Hyper_Parameters(class Neural_Network *const ptr_Neural_Network_received);
        bool Initialize__Hyper_Parameter(std::tuple<int, size_t, T, T, T, T> &ref_hyperparameter_tuple_received, class Neural_Network *const ptr_Neural_Network_received);
        bool Shuffle__Hyper_Parameter(void);
        bool Feed__Hyper_Parameter(void);
        bool Deinitialize__OpenMP(void);
        
    #if defined(COMPILE_x86)
        bool Feed__Hyper_Parameter(std::tuple<int, size_t, T, T, T, T> const ref_hyperparameter_tuple_received, class Neural_Network *const ptr_Neural_Network_received);
    #elif defined(COMPILE_x64)
        bool Feed__Hyper_Parameter(std::tuple<int, size_t, T, T, T, T> const &ref_hyperparameter_tuple_received, class Neural_Network *const ptr_Neural_Network_received);
    #endif
        
        std::string Get__ID_To_String(int const hyperparameter_id_received) const;

    private:
        bool Enable__OpenMP__Population(void);
        bool Disable__OpenMP__Population(void);

        bool Allocate__Population(size_t const population_size_received);
        bool Allocate__Thread(size_t const number_threads_received);
        bool Reallocate__Population(size_t const population_size_received);
        bool Reallocate__Thread(size_t const number_threads_received);
        bool Optimize__Loop(size_t const number_iterations_received,
                                        class Dataset_Manager<T> *const ptr_Dataset_Manager_received,
                                        class Neural_Network *const ptr_Neural_Network_received);
        bool Optimize__OpenMP(size_t const number_iterations_received,
                                             class Dataset_Manager<T> *const ptr_Dataset_Manager_received,
                                             class Neural_Network *const ptr_Neural_Network_received);
        bool Evaluation__Loop(class Dataset_Manager<T> *const ptr_Dataset_Manager_received);
        bool Evaluation__OpenMP(class Dataset_Manager<T> *const ptr_Dataset_Manager_received);
        bool _use_OpenMP = false;
        bool _is_OpenMP_initialized = false;

        size_t _population_size = 0_zu;
        size_t _number_threads = 0_zu;
        size_t _cache_number_threads = 0_zu;
        
        double _population_gaussian_percent = 60.0; // Exploitation population. remaining exploration.
        double _percentage_maximum_thread_usage = 100.0;
        double _cache_maximum_threads_percent = 0.0;
        
        std::vector<std::tuple<int, size_t, T, T, T, T>> _vector_hyperparameters;

        /* std::tuple<
                            [0]: ID,
                            [1]: Layer/Unit index,
                            [2]: Value,
                            [3]: Value Minimum,
                            [4]: Value Maximum,
                            [5]: Variance
                                                > */
        std::tuple<int, size_t, T, T, T, T> *_ptr_selected_hyperparameter = nullptr;
        
        class Dataset_Manager<T> *p_ptr_array_dataset_manager = nullptr;

        class Neural_Network *p_ptr_array_individuals = nullptr;

        MyEA::Random::Integer<int> _Class_Generator_Random_Int; // Index generator.
        MyEA::Random::Floating<T> _Class_Generator_Random_Real;
        MyEA::Random::Gaussian<T> _Class_Generator_Random_Gaussian;
};

enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION : unsigned int
{
    TYPE_HYPERPARAMETER_OPTIMIZATION_NONE = 0u,
    TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH = 1u,
    TYPE_HYPERPARAMETER_OPTIMIZATION_LENGTH = 2u
};

static std::map<enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION, std::string> ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES = {{TYPE_HYPERPARAMETER_OPTIMIZATION_NONE, "NONE"},
                                                                                                                                                                                                                               {TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH, "Gaussian search"},
                                                                                                                                                                                                                               {TYPE_HYPERPARAMETER_OPTIMIZATION_LENGTH, "LENGTH"}};

template<typename T>
class Hyperparameter_Optimization
{
    protected:
        size_t p_number_hyper_optimization_iterations = 10_zu;
        size_t p_number_hyper_optimization_iterations_delay = 25_zu;
        size_t p_optimization_iterations_since_hyper_optimization = 0_zu;
        
    public:
        Hyperparameter_Optimization(void);
        virtual ~Hyperparameter_Optimization(void);
        
        void Reset(void);
        void Deallocate__Gaussian_Search(void);

        bool Set__Hyperparameter_Optimization(enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION const type_hyper_optimization_received);
        bool Set__Number_Hyperparameter_Optimization_Iterations(size_t const number_hyper_optimization_iterations_delay_received);
        bool Set__Number_Hyperparameter_Optimization_Iterations_Delay(size_t const number_hyper_optimization_iterations_delay_received);
        bool Get__Evaluation_Require(void) const;
        bool Optimize(class Dataset_Manager<T> *const ptr_Dataset_Manager_received, class Neural_Network *const ptr_Neural_Network_received);
        bool Evaluation(void);
        bool Evaluation(class Dataset_Manager<T> *const ptr_Dataset_Manager_received);
        bool User_Controls(void);
        bool User_Controls__Change__Hyperparameter_Optimization(void);
        bool Allocate__Gaussian_Search(void);
        bool Deallocate(void);

        T Optimization(class Dataset_Manager<T> *const ptr_Dataset_Manager_received, class Neural_Network *const ptr_Neural_Network_received);

        enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION Get__Hyperparameter_Optimization(void) const;

    private:
        bool _evaluation_require = false;

        class Gaussian_Search<T> *_ptr_Gaussian_Search = nullptr;

        enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION _type_hyperparameter_optimization = ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE;
};

template<typename T>
class Dataset_Cross_Validation_Hyperparameter_Optimization : public Dataset_Cross_Validation<T>, public Hyperparameter_Optimization<T>
{
    public:
        Dataset_Cross_Validation_Hyperparameter_Optimization(void);
        virtual ~Dataset_Cross_Validation_Hyperparameter_Optimization(void);
        
        virtual T Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received);
        virtual T Training_Loop(class Neural_Network *const ptr_Neural_Network_received);
        
        bool Deallocate(void);
};

struct Dataset_Parameters
{
    public:
        /* value_0:
                [-1]: User choose,
                [Mini-batch stochastic gradient descent]: shuffle,
                [Cross-validation]: shuffle */
        int value_0 = -1;
        /* value_1:
                [-1]: User choose,
                [Mini-batch stochastic gradient descent]: number_desired_data_per_batch,
                [Cross-validation]: number_k_fold */
        int value_1 = -1;
        /* value_2:
                [-1]: User choose,
                [Mini-batch stochastic gradient descent]: number_maximum_batch,
                [Cross-validation]: number_k_sub_fold */
        int value_2 = -1;
        /* value_3:
                [-1]: User choose,
                [Cross-validation, hyper-optimization]: Number hyperparameter optimization iteration(s) */
        int value_3 = -1;
        /* value_3:
                [-1]: User choose,
                [Cross-validation, hyper-optimization]: Number hyperparameter optimization iteration(s) delay */
        int value_4 = -1;
};

struct Dataset_Manager_Parameters
{
    public:
        /* Type storage:
                [-1]: User choose.
                [0]: Training.
                [1]: Training and testing.
                [2]: Training, validation and testing. */
        int type_storage = -1;
        /* Type training:
                [-1]: User choose.
                [0]: Batch gradient descent.
                [1]: Mini-batch.
                [2]: Cross-validation.
                [3]: Cross-validation, random search. */
        int type_training = -1;

        double percent_training_size = 0.0;
        double percent_validation_size = 0.0;

        struct Dataset_Parameters training_parameters;
};

struct Data_Accuracy
{
    Data_Accuracy(void) { }
    ~Data_Accuracy(void) { }
    
    struct Data_Accuracy& operator=(struct Data_Accuracy const &ref_source_Data_Accuracy_received)
    {
        if(&ref_source_Data_Accuracy_received != this)
        {
            this->desired_accuracy = ref_source_Data_Accuracy_received.desired_accuracy;
            
            this->ptr_array_desired_entries = ref_source_Data_Accuracy_received.ptr_array_desired_entries;
        }

        return(*this);
    }

    T_ desired_accuracy = 100_T;

    T_ *ptr_array_desired_entries = nullptr;
};

template<typename T>
class Dataset_Manager : public Dataset<T>, public Hyperparameter_Optimization<T>
{
    public:
        Dataset_Manager(void);
        Dataset_Manager(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received);
        virtual ~Dataset_Manager(void);

        void Testing_On_Storage(class Neural_Network *const ptr_Neural_Network_received);
        void Set__Evaluation(enum MyEA::Common::ENUM_TYPE_DATASET const type_evaluation_received);
        void Set__Desired_Optimization_Time_Between_Reports(double const desired_optimization_time_between_reports_received);
        void Optimization(struct MyEA::Common::While_Condition const &ref_while_condition_received,
                                  bool const save_neural_network_trainer_received,
                                  bool const save_neural_network_trained_received,
                                  T const desired_loss_received,
                                  std::string const &ref_path_net_trainer_neural_network_received,
                                  std::string const &ref_path_nn_trainer_neural_network_received,
                                  std::string const &ref_path_net_trained_neural_network_received,
                                  std::string const &ref_path_nn_trained_neural_network_received,
                                  class Neural_Network *&ptr_trainer_Neural_Network_received,
                                  class Neural_Network *&ptr_trained_Neural_Network_received);
    #if defined(COMPILE_COUT)
        #if defined(COMPILE_WINDOWS)
        void Optimization__Testing(bool const report_received,
                                                std::chrono::steady_clock::time_point &ref_time_start_received,
                                                std::chrono::steady_clock::time_point &ref_time_end_received,
                                                class Neural_Network *&ptr_trainer_Neural_Network_received);
        #elif defined(COMPILE_LINUX)
        void Optimization__Testing(bool const report_received,
                                                std::chrono::_V2::system_clock::time_point &ref_time_start_received,
                                                std::chrono::_V2::system_clock::time_point &ref_time_end_received,
                                                class Neural_Network *&ptr_trainer_Neural_Network_received);
        #endif
    #else
        void Optimization__Testing(bool const report_received, class Neural_Network *&ptr_trainer_Neural_Network_received);
    #endif
        void Deallocate__Storage(void);
        void Deallocate__Shutdown_Boolean(void);
        
        bool Set__Maximum_Data(size_t const number_examples_received);
        bool Allocate__Shutdown_Boolean(void);
        bool Reallocate_Internal_Storage(void);
        bool Push_Back(T const *const ptr_array_inputs_received, T const *const ptr_array_outputs_received);
        bool Prepare_Storage(class Dataset<T> *const ptr_TrainingSet_received);
        bool Prepare_Storage(size_t const number_examples_training_received,
                                        size_t const number_examples_testing_received,
                                        class Dataset<T> *const ptr_TrainingSet_received,
                                        class Dataset<T> *const ptr_TestingSet_received);
        bool Prepare_Storage(size_t const number_examples_training_received,
                                        size_t const number_examples_validation_received,
                                        size_t const number_examples_testing_received,
                                        class Dataset<T> *const ptr_TrainingSet_received,
                                        class Dataset<T> *const ptr_ValidatingSet_received,
                                        class Dataset<T> *const ptr_TestingSet_received);
        bool Prepare_Storage(double const number_examples_percent_training_received,
                                        double const number_examples_percent_testing_received,
                                        class Dataset<T> *const ptr_TrainingSet_received,
                                        class Dataset<T> *const ptr_TestingSet_received);
        bool Prepare_Storage(double const number_examples_percent_training_received,
                                        double const number_examples_percent_validation_received,
                                        double const number_examples_percent_testing_received,
                                        class Dataset<T> *const ptr_TrainingSet_received,
                                        class Dataset<T> *const ptr_ValidatingSet_received,
                                        class Dataset<T> *const ptr_TestingSet_received);
        bool Initialize_Dataset(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                         enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_dataset_process_received,
                                         struct Dataset_Parameters const *const ptr_Dataset_Parameters_received = nullptr);
        bool Preparing_Dataset_Manager(struct Dataset_Manager_Parameters const *const ptr_Dataset_Manager_Parameters_received = nullptr);
        bool Reference(class Dataset_Manager<T> *const ptr_source_Dataset_Manager_received);
        bool Copy__Storage(class Dataset_Manager<T> const *const ptr_source_Dataset_Manager_received);
        bool User_Controls(void);
        bool User_Controls__Set__Maximum_Data(void);
        bool User_Controls__Type_Evaluation(void);
        bool User_Controls__Type_Metric(void);
        bool User_Controls__Optimization_Processing_Parameters(void);
        bool User_Controls__Optimization_Processing_Parameters__Batch(void);
        bool User_Controls__Optimization_Processing_Parameters__Mini_Batch(void);
        bool User_Controls__Optimization_Processing_Parameters__Cross_Validation(void);
        bool User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search(void);
        bool User_Controls__Optimization(class Neural_Network *&ptr_trainer_Neural_Network_received, class Neural_Network *&ptr_trained_Neural_Network_received);
        bool Assign_Shutdown_Block(class MyEA::Capturing::Shutdown &shutdown_module);
        bool Get__On_Shutdown(void) const;
        bool Get__Dataset_In_Equal_Less_Holdout_Accepted(void) const;
        bool Use__Metric_Loss(void) const;
        virtual bool Deallocate(void);

        virtual T Training(class Neural_Network *const ptr_Neural_Network_received);
        T Optimize(class Neural_Network *const ptr_Neural_Network_received);
        T Type_Testing(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, class Neural_Network *const ptr_Neural_Network_received);
        std::pair<T, T> Type_Update_Batch_Normalization(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, class Neural_Network *const ptr_Neural_Network_received);
        T Evaluate(class Neural_Network *const ptr_Neural_Network_received);
        T Get__Minimum_Loss_Holdout_Accepted(void) const;

        enum MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE Get__Type_Storage(void) const;
        
        class Dataset<T> *Allocate__Dataset(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_dataset_process_received, enum MyEA::Common::ENUM_TYPE_DATASET const type_data_received);
        class Dataset<T> *Get__Dataset_At(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const;

    #if defined(COMPILE_UI)
        void Set__Plot__Loss(bool const use_plot_received);
        void Set__Plot__Accuracy(bool const use_plot_received);
        void Set__Plot__Output(bool const use_plot_received);
        void Set__Plot__Output__Is_Image(bool const is_plot_image_received);
        void Set__Maximum_Ploted_Examples(size_t const number_examples_received);
        
        bool Set__Time_Delay_Ploted(size_t const time_delay_received);
        bool Get__Plot__Loss(void) const;
        bool Get__Plot__Accuracy(void) const;
        bool Get__Plot__Output(void) const;
        bool User_Controls__Set__Maximum_Ploted_Example(void);
        bool User_Controls__Set__Time_Delay_Ploted(void);
        bool Plot__Dataset_Manager(int const input_index_received, enum ENUM_TYPE_INPUT const type_input_received);
        bool Plot__Dataset_Manager(enum ENUM_TYPE_INPUT const type_input_received);
        bool Plot__Dataset_Manager(void);
        bool Plot__Dataset_Manager__Pre_Training(class Neural_Network *const ptr_Neural_Network_received);
        bool Plot__Neural_Network(class Neural_Network *const ptr_Neural_Network_received);

        size_t Get__Maximum_Ploted_Examples(void) const;
        size_t Get__Time_Delay_Ploted(void) const;
    #endif

    #if defined(COMPILE_CUDA)
        void Deallocate_CUDA(void);
        
        bool Initialize__CUDA(void);
        bool Initialize_Dataset_CUDA(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received);

        class CUDA_Dataset_Manager<T>* Get__CUDA(void);
    #endif

        enum MyEA::Common::ENUM_TYPE_DATASET Get__Type_Dataset_Evaluation(void) const;

    private:
        bool _reference = false;
        bool _dataset_in_equal_less_holdout_accepted = true;
        bool _use_metric_loss = true;
        std::atomic<bool> *_ptr_shutdown_boolean = nullptr;

        size_t _maximum_examples = 0_zu;

        T _minimum_loss_holdout_accepted = (std::numeric_limits<ST_>::max)();

        double _size_dataset_training__percent = 60.0;
        double _size_dataset_validation__percent = 20.0;
        double _size_dataset_testing__percent = 20.0;
        double _desired_optimization_time_between_reports = 1.0; // Seconds

        enum MyEA::Common::ENUM_TYPE_DATASET _type_evaluation = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION;
        enum MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE _type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;

        class Dataset<T> **_ptr_array_ptr_Dataset = nullptr;
        
    #if defined(COMPILE_UI)
        bool _use_plot_loss = true;
        bool _use_plot_accuracy = true;
        bool _is_plot_output_image = false;
        bool _use_plot_output = true;

        size_t _maximum_ploted_examples = 0_zu;
        size_t _time_delay_ploted = 0_zu;
    #endif

    #if defined(COMPILE_CUDA)
       class CUDA_Dataset_Manager<T> *_ptr_CUDA_Dataset_Manager = NULL;
    #endif
};

