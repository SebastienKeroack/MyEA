#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    #include <iterator> 
#endif

namespace MyEA::Memory
{
    template<class T>
    void Copy(T const *ptr_array_source_received,
              T const *ptr_array_last_source_received,
              T *ptr_array_destination_received)
    {
    #if defined(_DEBUG) || defined(COMPILE_DEBUG) && defined(COMPILE_WINDOWS)
            std::copy(ptr_array_source_received,
                      ptr_array_last_source_received,
                      stdext::checked_array_iterator<T*>(ptr_array_destination_received, static_cast<size_t>(ptr_array_last_source_received - ptr_array_source_received)));
    #else
            std::copy(ptr_array_source_received,
                      ptr_array_last_source_received,
                      ptr_array_destination_received);
    #endif
    }
    
    template<class T>
    void Copy_Loop(T const *ptr_array_source_received,
                   T const *ptr_array_last_source_received,
                   T *ptr_array_destination_received)
    {
        while(ptr_array_source_received != ptr_array_last_source_received)
        {
            *ptr_array_destination_received = *ptr_array_source_received;

            ++ptr_array_source_received;
            ++ptr_array_destination_received;
        }
    }
    
    template<class T>
    void Fill(T *ptr_array_received,
              T *ptr_array_last_received,
              T const value_received)
    {
        std::fill(ptr_array_received,
                  ptr_array_last_received,
                  value_received);
    }
    
    template<class T>
    void Fill_Nullptr(T *ptr_array_received, T *ptr_array_last_received)
    {
        for(; ptr_array_received != ptr_array_last_received; ++ptr_array_received) { *ptr_array_received = nullptr; }
    }

    template<class T>
    T* reallocate(T *ptr_array_received,
                  size_t const size_t_new_received,
                  size_t const size_t_old_received,
                  bool const memcpy_received)
    {
        if(ptr_array_received == NULL) { return(NULL); }
        else if(size_t_new_received == size_t_old_received) { return(ptr_array_received); }
            
        T *tmp_ptr_array_T;

        if(memcpy_received && size_t_old_received != 0_zu)
        {
            tmp_ptr_array_T = static_cast<T*>(malloc(size_t_new_received));

            if(size_t_old_received < size_t_new_received)
            {
                memset(tmp_ptr_array_T + (size_t_old_received / sizeof(T)),
                       0,
                       size_t_new_received - size_t_old_received);

                memcpy(tmp_ptr_array_T,
                       ptr_array_received,
                       size_t_old_received);
            }
            else
            {
                memcpy(tmp_ptr_array_T,
                       ptr_array_received,
                       size_t_new_received);
            }

            free(ptr_array_received);
            ptr_array_received = NULL;
        }
        else
        {
            free(ptr_array_received);
            ptr_array_received = NULL;
                    
            tmp_ptr_array_T = static_cast<T*>(malloc(size_t_new_received));
            memset(tmp_ptr_array_T,
                   0,
                   size_t_new_received);
        }

        return(tmp_ptr_array_T);
    }
    
    template<class T>
    T* reallocate_cpp(T *ptr_array_received,
                      size_t const new_size_received,
                      size_t const old_size_received,
                      bool const memcpy_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(memcpy_received && old_size_received != 0_zu)
        {
            tmp_ptr_array_T= new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                Fill<T>(tmp_ptr_array_T + old_size_received,
                        tmp_ptr_array_T + new_size_received,
                        T(0));

                Copy<T>(ptr_array_received,
                        ptr_array_received + old_size_received,
                        tmp_ptr_array_T);
            }
            else
            {
                Copy<T>(ptr_array_received,
                        ptr_array_received + new_size_received,
                        tmp_ptr_array_T);
            }
            
            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T= new T[new_size_received];

            Fill<T>(tmp_ptr_array_T,
                    tmp_ptr_array_T + new_size_received,
                    T(0));
        }

        return(tmp_ptr_array_T);
    }

    template<class T>
    T* reallocate_objects_cpp(T *ptr_array_received,
                              size_t const new_size_received,
                              size_t const old_size_received,
                              bool const memcpy_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(memcpy_received && old_size_received != 0_zu)
        {
            tmp_ptr_array_T= new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                Copy<T>(ptr_array_received,
                        ptr_array_received + old_size_received,
                        tmp_ptr_array_T);
            }
            else
            {
                Copy<T>(ptr_array_received,
                        ptr_array_received + new_size_received,
                        tmp_ptr_array_T);
            }
            
            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T= new T[new_size_received];
        }

        return(tmp_ptr_array_T);
    }

    template<class T>
    T* reallocate_pointers_array_cpp(T *ptr_array_received,
                                     size_t const new_size_received,
                                     size_t const old_size_received,
                                     bool const memcpy_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(memcpy_received && old_size_received != 0_zu)
        {
            tmp_ptr_array_T= new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                Fill_Nullptr<T>(tmp_ptr_array_T + old_size_received, tmp_ptr_array_T + new_size_received);

                Copy_Loop<T>(ptr_array_received,
                             ptr_array_received + old_size_received,
                             tmp_ptr_array_T);
            }
            else
            {
                Copy_Loop<T>(ptr_array_received,
                             ptr_array_received + new_size_received,
                             tmp_ptr_array_T);
            }
            
            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T= new T[new_size_received];

            Fill_Nullptr<T>(tmp_ptr_array_T, tmp_ptr_array_T + new_size_received);
        }

        return(tmp_ptr_array_T);
    }
}