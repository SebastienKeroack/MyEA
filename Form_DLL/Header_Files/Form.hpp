#pragma once

#include <Tools/Configuration.hpp>

namespace MyEA
{
    namespace Form
    {
        DLL_API bool DLL_API API__Form__Is_Loaded(void);

        DLL_API void DLL_API API__Form__Neural_Network__Allocate(void);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Use_Datapoint_Training(bool const use_datapoint_training_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Initialize(unsigned int const type_chart_received, unsigned int const number_series_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Total_Means(unsigned int const total_means_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Reset(unsigned int const type_chart_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Rescale(unsigned int const type_chart_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Loss_Diff(unsigned int const index_series_received,
                                                                                                                           unsigned int const type_received,
                                                                                                                           double const x_received);

        DLL_API void DLL_API API__Form__Neural_Network__Chart_Add_Point(unsigned int const type_chart_received,
                                                                                                                           unsigned int const index_series_received,
                                                                                                                           unsigned int const type_loss_received,
                                                                                                                           double const x_received,
                                                                                                                           double const y_received);

        DLL_API void DLL_API API__Form__Neural_Network__Chart_Grid_Search_Add_Column(std::string const &ref_value_received);
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Grid_Search_Add_Row(unsigned int const cell_index_received, std::string const &ref_value_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Scale(unsigned int const type_chart_received,
                                                                                                                     unsigned int const index_series_received,
                                                                                                                     unsigned int const type_loss_received,
                                                                                                                     bool const scale_y_axis_received,
                                                                                                                     double const x_received,
                                                                                                                     double const y_received);
        
        DLL_API void DLL_API API__Form__Neural_Network__Deallocate(void);
        
        DLL_API bool DLL_API API__Form__Neural_Network__Get_Signal_Training_Stop(void);
        
        DLL_API bool DLL_API API__Form__Neural_Network__Get_Signal_Training_Menu(void);
        
        DLL_API bool DLL_API API__Form__Neural_Network__Reset_Signal_Training_Menu(void);
    }
}