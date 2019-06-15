#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Count_Identical_Outputs_Entries.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Count_Identical_Outputs_Entries(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::String::Get__Time() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Count Identical Outputs Entries").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE);
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name.c_str(),
                                 tmp_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_Neural_Network_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    size_t const tmp_number_outputs(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs());
    size_t tmp_number_examples,
              tmp_index;

    std::vector<T_> tmp_vector_identical_entries;

    switch(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Type_Storage())
    {
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
            tmp_vector_identical_entries.clear();

            tmp_number_examples = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Examples();

            for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
            {
                tmp_vector_identical_entries.push_back(MyEA::String::Cin_Real_Number<T_>(-1.0f,
                                                                                                                                   1.0f,
                                                                                                                                   MyEA::String::Get__Time() + ": Output " + std::to_string(tmp_index) + " to search (based on " + std::to_string(tmp_number_examples) + " data): "));
            }

            PRINT_FORMAT("%s: Found %zu identical output entrie(s)." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Identical_Outputs(tmp_vector_identical_entries));
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
            while(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to count how many outputs are identical (training dataset)?"))
            {
                tmp_vector_identical_entries.clear();
                
                tmp_number_examples = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Examples();

                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                {
                    tmp_vector_identical_entries.push_back(MyEA::String::Cin_Real_Number<T_>(-1.0f,
                                                                                                                                       1.0f,
                                                                                                                                       MyEA::String::Get__Time() + ": Output " + std::to_string(tmp_index) + " to search (based on " + std::to_string(tmp_number_examples) + " data): "));
                }

                PRINT_FORMAT("%s: Found %zu identical output entrie(s)." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Identical_Outputs(tmp_vector_identical_entries));
            }

            while(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to count how many outputs are identical (testing dataset)?"))
            {
                tmp_vector_identical_entries.clear();
                
                tmp_number_examples = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Get__Number_Examples();

                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                {
                    tmp_vector_identical_entries.push_back(MyEA::String::Cin_Real_Number<T_>(-1.0f,
                                                                                                                                       1.0f,
                                                                                                                                       MyEA::String::Get__Time() + ": Output " + std::to_string(tmp_index) + " to search (based on " + std::to_string(tmp_number_examples) + " data): "));
                }

                PRINT_FORMAT("%s: Found %zu identical output entrie(s)." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),  
                                         tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Get__Identical_Outputs(tmp_vector_identical_entries));
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
            while(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to count how many outputs are identical (training dataset)?"))
            {
                tmp_vector_identical_entries.clear();
                
                tmp_number_examples = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Examples();

                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                {
                    tmp_vector_identical_entries.push_back(MyEA::String::Cin_Real_Number<T_>(-1.0f,
                                                                                                                                       1.0f,
                                                                                                                                       MyEA::String::Get__Time() + ": Output " + std::to_string(tmp_index) + " to search (based on " + std::to_string(tmp_number_examples) + " data): "));
                }

                PRINT_FORMAT("%s: Found %zu identical output entrie(s)." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Identical_Outputs(tmp_vector_identical_entries));
            }

            while(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to count how many outputs are identical (validation dataset)?"))
            {
                tmp_vector_identical_entries.clear();
                
                tmp_number_examples = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Get__Number_Examples();

                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                {
                    tmp_vector_identical_entries.push_back(MyEA::String::Cin_Real_Number<T_>(-1.0f,
                                                                                                                                       1.0f,
                                                                                                                                       MyEA::String::Get__Time() + ": Output " + std::to_string(tmp_index) + " to search (based on " + std::to_string(tmp_number_examples) + " data): "));
                }

                PRINT_FORMAT("%s: Found %zu identical output entrie(s)." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Get__Identical_Outputs(tmp_vector_identical_entries));
            }

            while(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to count how many outputs are identical (testing dataset)?"))
            {
                tmp_vector_identical_entries.clear();
                
                tmp_number_examples = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Get__Number_Examples();

                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                {
                    tmp_vector_identical_entries.push_back(MyEA::String::Cin_Real_Number<T_>(-1.0f,
                                                                                                                                       1.0f,
                                                                                                                                       MyEA::String::Get__Time() + ": Output " + std::to_string(tmp_index) + " to search (based on " + std::to_string(tmp_number_examples) + " data): "));
                }

                PRINT_FORMAT("%s: Found %zu identical output entrie(s)." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Get__Identical_Outputs(tmp_vector_identical_entries));
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Type_Storage(),
                                     MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Type_Storage()].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}
