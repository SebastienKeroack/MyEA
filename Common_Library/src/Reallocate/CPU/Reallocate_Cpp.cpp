namespace MyEA::Memory::Cpp
{
    template<class T>
    void Fill_Nullptr(T *ptr_array_received, T const *const ptr_array_last_received)
    {
        while(ptr_array_received != ptr_array_last_received)
        {
            *ptr_array_received++ = nullptr;
        }
    }

    template<class T, bool CPY, bool SET>
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                if(SET)
                {
                    MyEA::Memory::Fill<T>(tmp_ptr_array_T + old_size_received,
                                          tmp_ptr_array_T + new_size_received,
                                          T(0));
                }

                MyEA::Memory::Copy<T>(ptr_array_received,
                                      ptr_array_received + old_size_received,
                                      tmp_ptr_array_T);
            }
            else
            {
                MyEA::Memory::Copy<T>(ptr_array_received,
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

            tmp_ptr_array_T = new T[new_size_received];

            if(SET)
            {
                MyEA::Memory::Fill<T>(tmp_ptr_array_T,
                                      tmp_ptr_array_T + new_size_received,
                                      T(0));
            }
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY>
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                MyEA::Memory::Copy<T>(ptr_array_received,
                                      ptr_array_received + old_size_received,
                                      tmp_ptr_array_T);
            }
            else
            {
                MyEA::Memory::Copy<T>(ptr_array_received,
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

            tmp_ptr_array_T = new T[new_size_received];
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY, bool SET>
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                if(SET) { Fill_Nullptr<T>(tmp_ptr_array_T + old_size_received, tmp_ptr_array_T + new_size_received); }

                // TODO: Check if std is compatible.
                MyEA::Memory::Copy<T, false>(ptr_array_received,
                                             ptr_array_received + old_size_received,
                                             tmp_ptr_array_T);
            }
            else
            {
                // TODO: Check if std is compatible.
                MyEA::Memory::Copy<T, false>(ptr_array_received,
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

            tmp_ptr_array_T = new T[new_size_received];

            if(SET) { Fill_Nullptr<T>(tmp_ptr_array_T, tmp_ptr_array_T + new_size_received); }
        }

        return(tmp_ptr_array_T);
    }
}