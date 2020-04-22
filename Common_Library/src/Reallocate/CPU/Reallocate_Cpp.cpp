/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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