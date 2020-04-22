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

void Update_Weight_QuickProp(class Neural_Network *ptr_Neural_Network_received,
                                                       size_t const number_examples_received,
                                                       unsigned int const first_weight_received,
                                                       unsigned int const past_end_received);
void Update_Weight_Batch(class Neural_Network *ptr_Neural_Network_received,
                                               size_t const number_examples_received,
                                               unsigned int const first_weight_received,
                                               unsigned int const past_end_received);
void Update_Weight_SARProp(class Neural_Network *ptr_Neural_Network_received,
                                                     unsigned int const epoch_received,
                                                     unsigned int const first_weight_received,
                                                     unsigned int const past_end_received);
