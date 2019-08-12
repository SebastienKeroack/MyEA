#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_STATE_PROPAGATION : unsigned int
        {
            TYPE_STATE_PROPAGATION_INFERENCE = 0u,
            TYPE_STATE_PROPAGATION_TRAINING = 1u,
            TYPE_STATE_PROPAGATION_UPDATE_BATCH_NORMALIZATION = 2u,
            TYPE_STATE_PROPAGATION_LENGTH = 3u
        };
        
        static std::map<enum ENUM_TYPE_STATE_PROPAGATION, std::string> ENUM_TYPE_STATE_PROPAGATION_NAMES = {{TYPE_STATE_PROPAGATION_INFERENCE, "Inference"},
                                                                                                                                                                                              {TYPE_STATE_PROPAGATION_TRAINING, "Training"},
                                                                                                                                                                                              {TYPE_STATE_PROPAGATION_UPDATE_BATCH_NORMALIZATION, "Update batch-normalization"},
                                                                                                                                                                                              {TYPE_STATE_PROPAGATION_LENGTH, "LENGTH"}};
    }
}
