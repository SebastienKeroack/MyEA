#include "stdafx.hpp"

#include <Arrays/CArrayObject.hpp>
#include <Expert/CExpert.hpp>

namespace MyEA
{
    namespace Common
    {
        CArrayObject::CArrayObject(void)
        {
        }

        CObject* CArrayObject::Get__At(const unsigned int index_received)
        {
            if(index_received < this->p_Ptr_data->size()) { return(this->p_Ptr_data->at(index_received)); } else { return(nullptr); }
        }

        void CArrayObject::Clear(void)
        {
            if(!this->p_Ptr_data->empty())
            {
                //for(int i(0); i < this->Get__Total(); ++i) { delete(this->p_Ptr_data->at(i); }
                this->p_Ptr_data->clear();
            }
        }

        const bool CArrayObject::Push_Back(CObject* ptr_CTrain_Data_received)
        {
            if(ptr_CTrain_Data_received != nullptr)
            {
                //p_datas[0] = 10;
                //p_data.push_back(10);
                //p_Ptr_data->push_back(CObject*(new CObject));
                return(true);
            } else { return(false); }
        }

        CArrayObject::~CArrayObject(void)
        {
            this->Clear();
            delete(p_Ptr_data);
        }
    }
}
