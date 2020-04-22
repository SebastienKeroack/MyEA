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

// Standard.
#include <sstream>
#include <iomanip>
#include <vector>

// This.
#include <Configuration/Configuration.hpp>
#include <Enums/Enum_Type_String_Format.hpp>

#define STD_NEW_LINE std::cout << std::endl;
#define NEW_LINE "\r\n"

#if defined(COMPILE_WINDOWS)
    #define ESCAPE_FILE "\\"
    #define STD_STRING std::wstring
    #define STD_CIN std::wcin
    #define STD_COUT std::wcout
    #define Lw(string_received) wchar_t(string_received)
#elif defined(COMPILE_LINUX)
    #define ESCAPE_FILE "/"
    #define STD_STRING std::string
    #define STD_CIN std::cin
    #define STD_COUT std::cout
    #define Lw(string_received) char_t(string_received)
#endif

constexpr
std::streamsize operator ""_ss(unsigned long long int variable_to_size_t_received) // http://en.cppreference.com/w/cpp/language/user_literal
{
    return(static_cast<std::streamsize>(variable_to_size_t_received));
}

namespace MyEA::String
{
    void Print(char const *ptr_fmt_received, ...);

    void Print_With_Prefix(std::string const &ref_prefix_received, char const *ptr_fmt_received, ...);

    #define Error(fmt, ...) Print_With_Prefix(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", ERROR: ", fmt, ##__VA_ARGS__)

    template <typename T>
    bool Parse_Number(char *&ptr_array_characters,
                      char *const ptr_last_character,
                      T &ref_output);

    bool Read_Stream_Block(size_t &ref_block_size_received,
                           size_t const desired_block_size_received,
                           size_t const step_block_size_received,
                           std::vector<char> &ref_vector_buffers_received,
                           std::ifstream &ref_ifstream_received,
                           char const until_reach_received = 0x00);

    template <typename T>
    bool Read_Stream_Block_And_Parse_Number(char *&ptr_array_characters_received,
                                            char *&ptr_last_character_received,
                                            size_t &ref_block_size_received,
                                            size_t const desired_block_size_received,
                                            size_t const step_block_size_received,
                                            T &ref_output_received,
                                            std::vector<char> &ref_vector_buffers_received,
                                            std::ifstream &ref_ifstream_received,
                                            char const until_reach_received = 0x00);

    template <typename T>
    T Cin_Number(T const minimum_number_received,
                 T const maximum_number_received,
                 std::string const &ref_prefix_received = "Number: ");

    template <typename T>
    T Cin_Number(T const minimum_number_received, std::string const &ref_prefix_received = "Number: ");

    template <typename T>
    T Cin_Real_Number(T const minimum_real_number_received,
                      T const maximum_real_number_received,
                      std::string const &ref_prefix_received = "Number: ");

    template <typename T>
    T Cin_Real_Number(T const minimum_real_number_received, std::string const &ref_prefix_received = "Number: ");

    bool Accept(std::string const &ref_prefix_received);

    std::string System_Command(char const *const tmp_ptr_command_received);

    std::string To_Upper(std::string string_received);

    template<class T, enum ENUM_TYPE_STRING_FORMAT TYPE = ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>
    std::string To_string(T const value_received, unsigned int const number_precision_received = 16u)
    {
        std::ostringstream tmp_value_return;

        switch(TYPE)
        {
            case ENUM_TYPE_STRING_FORMAT::FIXED:        tmp_value_return << std::fixed        << std::setprecision(number_precision_received) << value_received; break;
            case ENUM_TYPE_STRING_FORMAT::SCIENTIFIC:   tmp_value_return << std::scientific   << std::setprecision(number_precision_received) << value_received; break;
            case ENUM_TYPE_STRING_FORMAT::HEXFLOAT:     tmp_value_return << std::hexfloat     << std::setprecision(number_precision_received) << value_received; break;
            case ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT: tmp_value_return << std::defaultfloat << std::setprecision(number_precision_received) << value_received; break;
            default:
                Error("The `%s` dialog box type is not supported in the switch.", ENUM_TYPE_STRING_FORMAT_NAMES[TYPE].c_str());
                    return("");
        }

        return(tmp_value_return.str());
    }
}

