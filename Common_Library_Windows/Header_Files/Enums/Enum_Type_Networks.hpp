#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_NETWORKS : unsigned int
        {
            TYPE_NETWORK_NONE = 0u,
            TYPE_NETWORK_AUTOENCODER = 1u,
            TYPE_NETWORK_CONVOLUTIONAL = 2u,
            TYPE_NETWORK_FEEDFORWARD = 3u,
            TYPE_NETWORK_RECURRENT = 4u,
            TYPE_NETWORK_LENGTH = 5u
        };

        static std::map<enum ENUM_TYPE_NETWORKS, std::string> ENUM_TYPE_NETWORKS_NAMES = {{TYPE_NETWORK_NONE, "NONE"},
                                                                                                                                                               {TYPE_NETWORK_AUTOENCODER, "Autoencoder"},
                                                                                                                                                               {TYPE_NETWORK_CONVOLUTIONAL, "[x] Convolutional"},
                                                                                                                                                               {TYPE_NETWORK_FEEDFORWARD, "Feedforward"},
                                                                                                                                                               {TYPE_NETWORK_RECURRENT, "Recurrent"},
                                                                                                                                                               {TYPE_NETWORK_LENGTH, "LENGTH"}};
    }
}
