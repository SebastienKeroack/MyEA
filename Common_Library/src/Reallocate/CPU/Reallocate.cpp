// Standard.
#if defined(_DEBUG) || defined(COMPILE_DEBUG) && defined(COMPILE_WINDOWS)
    #include <iterator> 
#endif

namespace MyEA::Memory
{
    template<class T, bool STD>
    void Copy(T const *ptr_array_source_received,
              T const *ptr_array_last_source_received,
              T *ptr_array_destination_received)
    {
        if(STD)
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
        else
        {
            while(ptr_array_source_received != ptr_array_last_source_received)
            {
                *ptr_array_destination_received++ = *ptr_array_source_received++;
            }
        }
    }
    
    template<class T, bool STD>
    void Fill(T *ptr_array_received,
              T *const ptr_array_last_received,
              T const value_received)
    {
        if(STD)
        {
            std::fill(ptr_array_received,
                      ptr_array_last_received,
                      value_received);
        }
        else
        {
            while(ptr_array_received != ptr_array_last_received)
            {
                *ptr_array_received++ = value_received;
            }
        }
    }
}