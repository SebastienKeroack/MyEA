#pragma once

#include <CObject.hpp>

namespace MyEA
{
    namespace Common
    {
        class CArray : public CObject
        {
            protected:
                int p_data_total;
                int p_data_max;

            public:
                CArray(void);
                ~CArray(void);

                // [     GET      ]
                const int Get__Total(void) const { return(p_data_total); }
                const int Get__Available(void) { return(p_data_max-p_data_total); }
                const int Get__Max(void) { return(p_data_max); }
                // ----- GET -----
        };
    }
}