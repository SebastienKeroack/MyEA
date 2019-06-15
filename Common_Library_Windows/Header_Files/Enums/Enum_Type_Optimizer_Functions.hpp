#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_OPTIMIZER_FUNCTIONS : unsigned int
        {
            TYPE_OPTIMIZER_NONE = 0u,
            TYPE_OPTIMIZER_ADABOUND = 1u, // https://openreview.net/forum?id=Bkg3g2R9FX
            TYPE_OPTIMIZER_ADAM = 2u,
            TYPE_OPTIMIZER_ADAMAX = 3u,
            TYPE_OPTIMIZER_AMSBOUND = 4u, // https://openreview.net/forum?id=Bkg3g2R9FX
            TYPE_OPTIMIZER_AMSGrad = 5u,
            TYPE_OPTIMIZER_GD = 6u,
            TYPE_OPTIMIZER_iRPROP_minus = 7u,
            TYPE_OPTIMIZER_iRPROP_plus = 8u,
            TYPE_OPTIMIZER_NosADAM = 9u, // https://arxiv.org/abs/1805.07557
            TYPE_OPTIMIZER_QUICKPROP = 10u,
            TYPE_OPTIMIZER_SARPROP = 11u,
            TYPE_OPTIMIZER_LENGTH = 12u
        };

        static std::map<enum ENUM_TYPE_OPTIMIZER_FUNCTIONS, std::string> ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES = {{TYPE_OPTIMIZER_NONE, "NONE"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_ADABOUND, "AdaBound"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_ADAM, "ADAM"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_ADAMAX, "AdaMax"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_AMSBOUND, "AMSBound"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_AMSGrad, "AMSGrad"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_GD, "Gradient descent"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_iRPROP_minus, "iRPROP-"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_iRPROP_plus, "iRPROP+"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_NosADAM, "Nostalgic Adam - HH"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_QUICKPROP, "QuickProp"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_SARPROP, "SARProp"},
                                                                                                                                                                                                    {TYPE_OPTIMIZER_LENGTH, "LENGTH"}};
    }
}
