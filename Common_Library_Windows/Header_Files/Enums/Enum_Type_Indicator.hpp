#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_INDICATORS
        {
            TYPE_iNONE = 0,
            TYPE_iNONE_NN = 1,
            TYPE_iNONE_RNN = 2,
            TYPE_iBEARSPOWER = 3,
            TYPE_iBEARSPOWER_NN = 4,
            TYPE_iBULLSPOWER = 5,
            TYPE_iBULLSPOWER_NN = 6,
            TYPE_iCANDLESTICK = 7,
            TYPE_iCCI = 8,
            TYPE_iCCI_NN = 9,
            TYPE_iFIBONACCI = 10,
            TYPE_iFIBONACCI_NN = 11,
            TYPE_iICHIMOKUKINKOHYO = 12,
            TYPE_iMACD = 13,
            TYPE_iMACD_NN = 14,
            TYPE_iMA = 15,
            TYPE_iMA_RNN_SIGN = 16,
            TYPE_iMA_RNN_PRICE = 17,
            TYPE_iMA_RNN_WPRICE = 18,
            TYPE_iRSI= 19,
            TYPE_iRSI_NN= 20,
            TYPE_iRVI = 21,
            TYPE_iRVI_NN = 22,
            TYPE_iSTOCHASTIC = 23,
            TYPE_iSTOCHASTIC_NN = 24,
            TYPE_AUTOENCODER = 25,
            TYPE_INDICATOR_LENGTH = 26
        };
        
        static std::map<enum ENUM_TYPE_INDICATORS, std::string> ENUM_TYPE_INDICATORS_NAMES = {{TYPE_iNONE, "TYPE_iNONE"},
                                                                                                                                                                {TYPE_iNONE_NN, "TYPE_iNONE_NN"},
                                                                                                                                                                {TYPE_iNONE_RNN, "TYPE_iNONE_RNN"},
                                                                                                                                                                {TYPE_iBEARSPOWER, "TYPE_iBEARSPOWER"},
                                                                                                                                                                {TYPE_iBEARSPOWER_NN, "TYPE_iBEARSPOWER_NN"},
                                                                                                                                                                {TYPE_iBULLSPOWER, "TYPE_iBULLSPOWER"},
                                                                                                                                                                {TYPE_iBULLSPOWER_NN, "TYPE_iBULLSPOWER_NN"},
                                                                                                                                                                {TYPE_iCANDLESTICK, "TYPE_iCANDLESTICK"},
                                                                                                                                                                {TYPE_iCCI, "TYPE_iCCI"},
                                                                                                                                                                {TYPE_iCCI_NN, "TYPE_iCCI_NN"},
                                                                                                                                                                {TYPE_iFIBONACCI, "TYPE_iFIBONACCI"},
                                                                                                                                                                {TYPE_iFIBONACCI_NN, "TYPE_iFIBONACCI_NN"},
                                                                                                                                                                {TYPE_iICHIMOKUKINKOHYO, "TYPE_iICHIMOKUKINKOHYO"},
                                                                                                                                                                {TYPE_iMACD, "TYPE_iMACD"},
                                                                                                                                                                {TYPE_iMACD_NN, "TYPE_iMACD_NN"},
                                                                                                                                                                {TYPE_iMA, "TYPE_iMA"},
                                                                                                                                                                {TYPE_iMA_RNN_SIGN, "TYPE_iMA_RNN_SIGN"},
                                                                                                                                                                {TYPE_iMA_RNN_PRICE, "TYPE_iMA_RNN_PRICE"},
                                                                                                                                                                {TYPE_iMA_RNN_WPRICE, "TYPE_iMA_RNN_WPRICE"},
                                                                                                                                                                {TYPE_iRSI, "TYPE_iRSI"},
                                                                                                                                                                {TYPE_iRSI_NN, "TYPE_iRSI_NN"},
                                                                                                                                                                {TYPE_iRVI, "TYPE_iRVI"},
                                                                                                                                                                {TYPE_iRVI_NN, "TYPE_iRVI_NN"},
                                                                                                                                                                {TYPE_iSTOCHASTIC, "TYPE_iSTOCHASTIC"},
                                                                                                                                                                {TYPE_iSTOCHASTIC_NN, "TYPE_iSTOCHASTIC_NN"},
                                                                                                                                                                {TYPE_AUTOENCODER, "TYPE_AUTOENCODER"},
                                                                                                                                                                {TYPE_INDICATOR_LENGTH, "LENGTH"}};
    }
}