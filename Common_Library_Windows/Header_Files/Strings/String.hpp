#pragma once

#include <Configuration/Configuration.hpp>

#include <regex>
#include <sstream>
#include <iomanip>

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
    enum ENUM_TYPE_MANIPULATOR_STRING : int
    {
        TYPE_MANIPULATOR_STRING_FIXED        = 0,
        TYPE_MANIPULATOR_STRING_SCIENTIFIC   = 1,
        TYPE_MANIPULATOR_STRING_HEXFLOAT     = 2,
        TYPE_MANIPULATOR_STRING_DEFAULTFLOAT = 3
    };

    void Print(char const *ptr_fmt_received, ...);

    void Print_With_Prefix(std::string const &ref_prefix_received, char const *ptr_fmt_received, ...);

    void Find_And_Replace(std::string       &ref_source,
                          std::string const &ref_find,
                          std::string const &ref_replace);

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

    bool NoOrYes(std::string const &ref_prefix_received);

    bool Regex_Read_Input(int &ref_result_received,
                          std::string const &ref_line_received,
                          std::regex const &ref_regex_received);

    bool Regex_Read_Input(float &ref_result_received,
                          std::string const &ref_line_received,
                          std::regex const &ref_regex_received);

    std::string Execute_Command(char const *const tmp_ptr_command_received);

#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    std::string Get__Time(std::string format_received = "", bool const use_local_time_received = true);
#else // COMPILE_DEBUG
    std::string Get__Time(std::string format_received = "", bool const use_local_time_received = false);
#endif // COMPILE_DEBUG

    std::string Get__Time_Elapse(double const time_elapse_received);

    std::string To_Upper(std::string string_to_uppercase_received);

    template<class T, enum MyEA::String::ENUM_TYPE_MANIPULATOR_STRING TYPE>
    std::string To_string(T const value_received, unsigned int const number_precision_received = 16u)
    {
        std::ostringstream tmp_value_return;

        switch(TYPE)
        {
            case MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED:        tmp_value_return << std::fixed        << std::setprecision(number_precision_received) << value_received; break;
            case MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_SCIENTIFIC:   tmp_value_return << std::scientific   << std::setprecision(number_precision_received) << value_received; break;
            case MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_HEXFLOAT:     tmp_value_return << std::hexfloat     << std::setprecision(number_precision_received) << value_received; break;
            case MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT: tmp_value_return << std::defaultfloat << std::setprecision(number_precision_received) << value_received; break;
            default: tmp_value_return << std::setprecision(number_precision_received) << value_received; break;
        }

        return(tmp_value_return.str());
    }

    #define Error(fmt, ...) Print_With_Prefix(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", ERROR: ", fmt, ##__VA_ARGS__)
}

