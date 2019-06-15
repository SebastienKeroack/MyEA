#pragma once

#include <Tools/Configuration.hpp>

namespace Memory
{
    template<class T>
    void Copy(T const *ptr_array_source_received,
                   T const *ptr_array_last_source_received,
                   T *ptr_array_destination_received);

    template<class T>
    void Copy_Loop(T const *ptr_array_source_received,
                            T const *ptr_array_last_source_received,
                            T *ptr_array_destination_received);
    
    template<class T>
    void Fill(T *ptr_array_received,
                T *ptr_array_last_received,
                T const value_received);

    template<class T>
    void Fill_Nullptr(T *ptr_array_received, T *ptr_array_last_received);

    template<class T>
    T* reallocate(T *ptr_array_received,
                       size_t const size_t_new_received,
                       size_t const size_t_old_received,
                       bool const memcpy_received = true);

    template<class T>
    T* reallocate_cpp(T *ptr_array_received,
                              size_t const new_size_received,
                              size_t const old_size_received,
                              bool const memcpy_received = true);

    template<class T>
    T* reallocate_objects_cpp(T *ptr_array_received,
                                          size_t const new_size_received,
                                          size_t const old_size_received,
                                          bool const memcpy_received = true);

    template<class T>
    T* reallocate_pointers_array_cpp(T *ptr_array_received,
                                                    size_t const new_size_received,
                                                    size_t const old_size_received,
                                                    bool const memcpy_received = true);
}

#include <../Source_Files/Tools/Reallocate_cpp.cpp>
