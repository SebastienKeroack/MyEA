#pragma once

#include <vector>
#include <memory>

#include <Arrays/CArray.hpp>

namespace MyEA
{
    namespace Common
    {
        class CArrayObject : public CArray
        {
            protected:
                int p_datas[5];
                std::vector<int> p_data;
                std::vector<CObject*>* p_Ptr_data = nullptr;

            public:
                CArrayObject(void);
                ~CArrayObject(void);

                // [     GET      ]
                CObject* Get__At(const unsigned int index_received);
                // ----- GET -----

                void Clear(void);

                const bool Push_Back(CObject* ptr_CTrain_Data_received);
        };
    }
}
