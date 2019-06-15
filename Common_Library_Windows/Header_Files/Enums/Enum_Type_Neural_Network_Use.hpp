#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_NEURAL_NETWORK_USE : unsigned int
        {
            TYPE_NEURAL_NETWORK_ALL = 0u,
            TYPE_NEURAL_NETWORK_TRAINER = 1u,
            TYPE_NEURAL_NETWORK_TRAINED = 2u,
            TYPE_NEURAL_NETWORK_COMPETITOR = 3u,
            TYPE_NEURAL_NETWORK_LENGTH = 4u
        };

        static std::map<enum ENUM_TYPE_NEURAL_NETWORK_USE, std::string> ENUM_TYPE_NEURAL_NETWORK_USE_NAMES = {{TYPE_NEURAL_NETWORK_ALL, "ALL"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_TRAINER, "Trainer"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_TRAINED, "Trained"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_COMPETITOR, "Competitor"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_LENGTH, "LENGTH"}};
    }
}