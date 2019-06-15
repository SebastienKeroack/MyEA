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
