#pragma once

#include <Neural_Network/Neural_Network_Manager.hpp>

template<typename T>
struct Dropout_Initializer__LW
{
    bool operator==(struct Dropout_Initializer__LW<T> const &ref_Dropout_Initializer__LW) const;

    size_t layer_index = 0u;

    T value[2u] = {0};

    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;
};

template<typename T>
struct Dropout_Initializer__Arguments
{
    size_t layer_index = 0_zu;

    T minimum_value[2u] = {0};
    T maximum_value[2u] = {0};
    T step_size[2u] = {0};

    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT type_layer_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;
};

template<typename T>
class Grid_Search
{
    protected:
        template<typename O> void Push_Back(std::vector<O> &ref_vector_received,
                                                                    O &ref_object_received,
                                                                    bool const allow_duplicate_received = true);

        std::vector<T> p_vector_Weight_Decay;
        std::vector<T> p_vector_Max_Norm_Constraints;
        std::vector<T> p_vector_Normalization_Momentum_Average;

        std::vector<std::vector<struct Dropout_Initializer__LW<T>>> p_vector_layers_Dropout;

    public:
        Grid_Search(void);
        ~Grid_Search(void);

        void Shuffle(void);
        void DataGridView_Initialize_Columns(void);
        void DataGridView_Add_Iteration(size_t const hyper_parameters_index_received, class Neural_Network *const ptr_Neural_Network_received);

        bool Input(class MyEA::Neural_Network::Neural_Network_Manager &ref_Neural_Network_Manager);
        bool Set__Maximum_Iterations(size_t const maximum_iterations_received);
        bool Set__Use__Shuffle(bool const use_shuffle_received);
        bool Push_Back__Weight_Decay(T const minimum_value_received,
                                                          T const maxium_value_received,
                                                          T const step_size_received,
                                                          bool const allow_duplicate_received = true);
        bool Push_Back__Max_Norm_Constraints(T const minimum_value_received,
                                                                      T const maxium_value_received,
                                                                      T const step_size_received,
                                                                      bool const allow_duplicate_received = true);
        bool Push_Back__Normalization_Momentum_Average(T const minimum_value_received,
                                                                                       T const maxium_value_received,
                                                                                       T const step_size_received,
                                                                                       bool const allow_duplicate_received = true);
        bool Push_Back__Dropout(struct Dropout_Initializer__Arguments<T> Dropout_Initializer__Arguments_received, bool const allow_duplicate_received = true);
        bool Update_Tree(void);
        bool User_Controls(void);
        bool User_Controls__Shuffle(void);
        bool Search(class MyEA::Neural_Network::Neural_Network_Manager &ref_Neural_Network_Manager_received);
        bool Feed_Hyper_Parameters(size_t const hyper_parameters_index_received, class Neural_Network *const ptr_Neural_Network_received);
        bool Allocate__Shutdown_Boolean(void);
        bool Assign_Shutdown_Block(class Shutdown_Block &ref_Shutdown_Block_received);
        bool Get__On_Shutdown(void) const;
        
    private:
        void Deallocate__Stochastic_Index(void);

        bool Allocate__Stochastic_Index(void);
        bool Reallocate__Stochastic_Index(size_t const total_iterations_received);
        bool _use_shuffle = false;
        std::atomic<bool> *_ptr_shutdown_boolean = nullptr;
        
        size_t _maximum_iterations = 1u;
        
        size_t *_ptr_array_stochastic_index = nullptr;
        size_t _total_iterations = 0_zu;
        
        std::vector<size_t> _vector_Tree;

        class MyEA::Common::Class_Generator_Random_Int<size_t> _Generator_Random_Integer;
};
