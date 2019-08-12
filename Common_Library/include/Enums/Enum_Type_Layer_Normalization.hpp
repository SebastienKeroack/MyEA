#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_LAYER_NORMALIZATION : unsigned int
        {
            TYPE_LAYER_NORMALIZATION_NONE = 0u,
            TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION = 1u,
            TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION = 2u,
            TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION = 3u,
            TYPE_LAYER_NORMALIZATION_STREAMING_NORMALIZATION = 4u,
            TYPE_LAYER_NORMALIZATION_LENGTH = 5u
        };

        static std::map<enum ENUM_TYPE_LAYER_NORMALIZATION, std::string> ENUM_TYPE_LAYER_NORMALIZATION_NAMES = {{TYPE_LAYER_NORMALIZATION_NONE, "NONE"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION, "Batch normalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION, "Batch renormalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION, "[x] Ghost batch normalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_STREAMING_NORMALIZATION, "[x] Streaming normalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_LENGTH, "LENGTH"}};

    }
}