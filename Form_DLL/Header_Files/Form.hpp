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

#pragma once

#include <Configuration/Configuration.hpp>

namespace MyEA
{
    namespace Form
    {
        DLL_API bool API__Form__Is_Loaded(void);

        DLL_API void API__Form__Neural_Network__Allocate(void);
        
        DLL_API void API__Form__Neural_Network__Chart_Use_Datapoint_Training(bool const use_datapoint_training_received);
        
        DLL_API void API__Form__Neural_Network__Chart_Initialize(unsigned int const type_chart_received, unsigned int const number_series_received);
        
        DLL_API void API__Form__Neural_Network__Chart_Total_Means(unsigned int const total_means_received);
        
        DLL_API void API__Form__Neural_Network__Chart_Reset(unsigned int const type_chart_received);
        
        DLL_API void API__Form__Neural_Network__Chart_Rescale(unsigned int const type_chart_received);
        
        DLL_API void API__Form__Neural_Network__Chart_Loss_Diff(unsigned int const index_series_received,
                                                                                                                           unsigned int const type_received,
                                                                                                                           double const x_received);

        DLL_API void API__Form__Neural_Network__Chart_Add_Point(unsigned int const type_chart_received,
                                                                                                                           unsigned int const index_series_received,
                                                                                                                           unsigned int const type_loss_received,
                                                                                                                           double const x_received,
                                                                                                                           double const y_received);

        DLL_API void API__Form__Neural_Network__Chart_Grid_Search_Add_Column(std::string const &ref_value_received);
        DLL_API void API__Form__Neural_Network__Chart_Grid_Search_Add_Row(unsigned int const cell_index_received, std::string const &ref_value_received);
        
        DLL_API void API__Form__Neural_Network__Chart_Scale(unsigned int const type_chart_received,
                                                                                                                     unsigned int const index_series_received,
                                                                                                                     unsigned int const type_loss_received,
                                                                                                                     bool const scale_y_axis_received,
                                                                                                                     double const x_received,
                                                                                                                     double const y_received);
        
        DLL_API void API__Form__Neural_Network__Deallocate(void);
        
        DLL_API bool API__Form__Neural_Network__Get_Signal_Training_Stop(void);
        
        DLL_API bool API__Form__Neural_Network__Get_Signal_Training_Menu(void);
        
        DLL_API bool API__Form__Neural_Network__Reset_Signal_Training_Menu(void);
    }
}