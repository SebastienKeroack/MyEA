namespace MyEA::Memory::C
{
    template<class T, bool CPY, bool SET>
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_t_received,
                  size_t const old_size_t_received)
    {
        if(ptr_array_received == NULL) { return(NULL); }
        else if(new_size_t_received == old_size_t_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_t_received != 0_zu)
        {
            tmp_ptr_array_T = static_cast<T*>(malloc(new_size_t_received));

            if(old_size_t_received < new_size_t_received)
            {
                if(SET)
                {
                    memset(tmp_ptr_array_T + (old_size_t_received / sizeof(T)),
                           0,
                           new_size_t_received - old_size_t_received);
                }

                memcpy(tmp_ptr_array_T,
                       ptr_array_received,
                       old_size_t_received);
            }
            else
            {
                memcpy(tmp_ptr_array_T,
                       ptr_array_received,
                       new_size_t_received);
            }

            free(ptr_array_received);
            ptr_array_received = NULL;
        }
        else
        {
            free(ptr_array_received);
            ptr_array_received = NULL;

            tmp_ptr_array_T = static_cast<T*>(malloc(new_size_t_received));

            if(SET)
            {
                memset(tmp_ptr_array_T,
                       0,
                       new_size_t_received);
            }
        }

        return(tmp_ptr_array_T);
    }
}