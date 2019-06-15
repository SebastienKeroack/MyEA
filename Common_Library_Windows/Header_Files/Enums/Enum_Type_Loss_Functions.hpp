#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_LOSS_FUNCTIONS : unsigned int
        {
            TYPE_LOSS_FUNCTION_NONE = 0u,
            TYPE_LOSS_FUNCTION_BIT = 1u,
            TYPE_LOSS_FUNCTION_CROSS_ENTROPY = 2u, // Cross entropy.
            TYPE_LOSS_FUNCTION_L1 = 3u, // L1, Least absolute deviation.
            TYPE_LOSS_FUNCTION_L2 = 4u, // L2, Least squares error.
            TYPE_LOSS_FUNCTION_MAE = 5u, // Mean absolute error.
            TYPE_LOSS_FUNCTION_MAPE = 6u, // Mean absolute percentage error.
            TYPE_LOSS_FUNCTION_MASE_NON_SEASONAL = 7u, // Mean absolute scaled error, non seasonal time series.
            TYPE_LOSS_FUNCTION_MASE_SEASONAL = 8u, // Mean absolute scaled error, seasonal time series.
            TYPE_LOSS_FUNCTION_ME = 9u, // Mean error.
            TYPE_LOSS_FUNCTION_MSE = 10u, // Mean square error.
            TYPE_LOSS_FUNCTION_RMSE = 11u, // Root mean square error.
            TYPE_LOSS_FUNCTION_SMAPE = 12u, // Symmetric mean absolute percentage error.
            TYPE_LOSS_FUNCTION_LENGTH = 13u
        };
        
        static std::map<enum ENUM_TYPE_LOSS_FUNCTIONS, std::string> ENUM_TYPE_LOSS_FUNCTIONS_NAMES = {{TYPE_LOSS_FUNCTION_NONE, "NONE"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_BIT, "Bit"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_CROSS_ENTROPY, "Cross-entropy"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_L1, "[-] L1"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_L2, "L2"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_MAE, "Mean absolute error"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_MAPE, "Mean absolute percentage error"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_MASE_NON_SEASONAL, "[x] Mean absolute scaled error, non seasonal time series"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_MASE_SEASONAL, "[x] Mean absolute scaled error, seasonal time series"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_ME, "[-] Mean error"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_MSE, "Mean square error"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_RMSE, "Root mean square error"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_SMAPE, "Symmetric mean absolute percentage error"},
                                                                                                                                                                                  {TYPE_LOSS_FUNCTION_LENGTH, "LENGTH"}};
        
        enum ENUM_TYPE_ACCURACY_FUNCTIONS : unsigned int
        {
            TYPE_ACCURACY_FUNCTION_NONE = 0u,
            TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY = 1u,
            TYPE_ACCURACY_FUNCTION_DIRECTIONAL = 2u,
            TYPE_ACCURACY_FUNCTION_DISTANCE = 3u,
            TYPE_ACCURACY_FUNCTION_R = 4u, // https://en.wikipedia.org/wiki/Correlation_coefficient | Pearson
            TYPE_ACCURACY_FUNCTION_SIGN = 5u,
            TYPE_ACCURACY_FUNCTION_LENGTH = 6u
        };
        
        static std::map<enum ENUM_TYPE_ACCURACY_FUNCTIONS, std::string> ENUM_TYPE_ACCURACY_FUNCTIONS_NAMES = {{TYPE_ACCURACY_FUNCTION_NONE, "NONE"},
                                                                                                                                                                                                      {TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY, "Cross-entropy"},
                                                                                                                                                                                                      {TYPE_ACCURACY_FUNCTION_DIRECTIONAL, "Directional"},
                                                                                                                                                                                                      {TYPE_ACCURACY_FUNCTION_DISTANCE, "Distance"},
                                                                                                                                                                                                      {TYPE_ACCURACY_FUNCTION_R, "R"},
                                                                                                                                                                                                      {TYPE_ACCURACY_FUNCTION_SIGN, "Sign"},
                                                                                                                                                                                                      {TYPE_ACCURACY_FUNCTION_LENGTH, "LENGTH"}};
    }
}
