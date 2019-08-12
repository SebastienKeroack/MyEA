#include "stdafx.hpp"

#if defined(COMPILE_LINUX)
    #include <cstring>
    #include <ctime>
#endif // COMPILE_LINUX

// This.
#include <Strings/String.hpp>
#include <Time/Time.hpp>

#include <boost/spirit/home/x3.hpp>

#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <regex>
#include <stdarg.h>

namespace MyEA::String
{
    template <typename T>
    bool Parse_Number(char *&ptr_array_characters_received,
                      char *const ptr_last_character_received,
                      T &ref_output_received)
    {
        auto tmp_Assign([&ref_output_received](auto &ref_ctx_received)
        {
            ref_output_received = _attr(ref_ctx_received);
        });

        auto tmp_Initializer([&tmp_Assign]()
        {
            if      constexpr (std::is_same<T, char              >::value) { return(boost::spirit::x3::char_     [tmp_Assign]); }
            else if constexpr (std::is_same<T, int               >::value) { return(boost::spirit::x3::int_      [tmp_Assign]); }
            else if constexpr (std::is_same<T, short             >::value) { return(boost::spirit::x3::short_    [tmp_Assign]); }
            else if constexpr (std::is_same<T, long              >::value) { return(boost::spirit::x3::long_     [tmp_Assign]); }
            else if constexpr (std::is_same<T, long long         >::value) { return(boost::spirit::x3::long_long [tmp_Assign]); }
            else if constexpr (std::is_same<T, unsigned short    >::value) { return(boost::spirit::x3::ushort_   [tmp_Assign]); }
            else if constexpr (std::is_same<T, unsigned int      >::value) { return(boost::spirit::x3::uint_     [tmp_Assign]); }
            else if constexpr (std::is_same<T, unsigned long     >::value) { return(boost::spirit::x3::ulong_    [tmp_Assign]); }
            else if constexpr (std::is_same<T, unsigned long long>::value) { return(boost::spirit::x3::ulong_long[tmp_Assign]); }
            else if constexpr (std::is_same<T, float             >::value) { return(boost::spirit::x3::float_    [tmp_Assign]); }
            else if constexpr (std::is_same<T, double            >::value) { return(boost::spirit::x3::double_   [tmp_Assign]); }
            else { throw(std::logic_error("NotImplementedException")); }
        });

        return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                               ptr_last_character_received,
                                               tmp_Initializer(),
                                               boost::spirit::x3::ascii::space));
    }

    bool Read_Stream_Block(size_t &ref_block_size_received,
                           size_t const desired_block_size_received,
                           size_t const step_block_size_received,
                           std::vector<char> &ref_vector_buffers_received,
                           std::ifstream &ref_ifstream_received,
                           char const until_reach_received)
    {
        if(ref_ifstream_received.eof()) { return(true); }

        // Current position.
        std::streampos const tmp_current_tellg(ref_ifstream_received.tellg());

        if(ref_ifstream_received.fail())
        {
            MyEA::String::Error("Can not gets the current position from the input stream.");

            return(false);
        }

        // Remaining characters based on current position.
        ref_ifstream_received.seekg(0, std::ios::end);

        if(ref_ifstream_received.fail())
        {
            MyEA::String::Error("Can not sets the position of the input stream.");

            return(false);
        }

        size_t const tmp_remaining_size(static_cast<size_t>(ref_ifstream_received.tellg() - tmp_current_tellg)),
                     tmp_block_size(tmp_remaining_size < desired_block_size_received ? tmp_remaining_size : desired_block_size_received);

        if(ref_ifstream_received.fail())
        {
            MyEA::String::Error("Can not gets the position from the input stream.");

            return(false);
        }

        // Return to the last position.
        ref_ifstream_received.seekg(tmp_current_tellg, std::ios::beg);

        if(ref_ifstream_received.fail())
        {
            MyEA::String::Error("Can not sets the position of the input stream to the beginning.");

            return(false);
        }

        // If not enough space in the buffer resize it.
        if(ref_vector_buffers_received.size() < tmp_block_size) { ref_vector_buffers_received.resize(tmp_block_size); }

        // Read block into buffers.
        ref_ifstream_received.read(&ref_vector_buffers_received[0], tmp_block_size);

        if(ref_ifstream_received.fail())
        {
            MyEA::String::Error("Can not read properly the file.");

            return(false);
        }

        // Store current block size.
        ref_block_size_received = tmp_block_size;

        // If we continue to read until reach a specific character.
        if(until_reach_received != 0x00
           &&
           ref_vector_buffers_received[tmp_block_size - 1] != until_reach_received
           &&
           ref_ifstream_received.eof() == false
           &&
           tmp_remaining_size != tmp_block_size)
        {
            // do while until reach.
            do
            {
                // If not enough space in the buffer resize it.
                if(ref_vector_buffers_received.size() < ref_block_size_received + 1) { ref_vector_buffers_received.resize(ref_block_size_received + 1 + step_block_size_received); }

                // Read character into buffer.
                ref_ifstream_received.read(&ref_vector_buffers_received[ref_block_size_received], 1);
            } while(ref_ifstream_received.eof() == false
                    &&
                    ref_ifstream_received.fail() == false
                    &&
                    ref_vector_buffers_received[ref_block_size_received++] != until_reach_received);

            if(ref_ifstream_received.fail())
            {
                MyEA::String::Error("Can not read properly the character (%c).", ref_vector_buffers_received[ref_block_size_received - 1]);

                return(false);
            }
        }

        return(true);
    }

    template <typename T>
    bool Read_Stream_Block_And_Parse_Number(char *&ptr_array_characters_received,
                                            char *&ptr_last_character_received,
                                            size_t &ref_block_size_received,
                                            size_t const desired_block_size_received,
                                            size_t const step_block_size_received,
                                            T &ref_output_received,
                                            std::vector<char> &ref_vector_buffers_received,
                                            std::ifstream &ref_ifstream_received,
                                            char const until_reach_received)
    {
        if(ptr_array_characters_received == ptr_last_character_received)
        {
            if(MyEA::String::Read_Stream_Block(ref_block_size_received,
                                               desired_block_size_received,
                                               step_block_size_received,
                                               ref_vector_buffers_received,
                                               ref_ifstream_received,
                                               until_reach_received) == false)
            {
                MyEA::String::Error("An error has been triggered from the `Read_Stream_Block(%zu, %zu, %zu, vector, ifstream, %s)` function.",
                                    ref_block_size_received,
                                    desired_block_size_received,
                                    step_block_size_received,
                                    until_reach_received);

                return(false);
            }

            ptr_array_characters_received = &ref_vector_buffers_received[0];
            ptr_last_character_received = ptr_array_characters_received + ref_block_size_received;
        }

        if(MyEA::String::Parse_Number<T>(ptr_array_characters_received,
                                         ptr_last_character_received,
                                         ref_output_received) == false)
        {
            MyEA::String::Error("An error has been triggered from the \"Parse_Number()\" function.");

            return(false);
        }

        return(true);
    }

    template bool Read_Stream_Block_And_Parse_Number<char              >(char *&, char *&, size_t &, size_t const, size_t const, char               &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<int               >(char *&, char *&, size_t &, size_t const, size_t const, int                &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<short             >(char *&, char *&, size_t &, size_t const, size_t const, short              &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<long              >(char *&, char *&, size_t &, size_t const, size_t const, long               &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<long long         >(char *&, char *&, size_t &, size_t const, size_t const, long long          &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<unsigned short    >(char *&, char *&, size_t &, size_t const, size_t const, unsigned short     &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<unsigned int      >(char *&, char *&, size_t &, size_t const, size_t const, unsigned int       &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<unsigned long     >(char *&, char *&, size_t &, size_t const, size_t const, unsigned long      &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<unsigned long long>(char *&, char *&, size_t &, size_t const, size_t const, unsigned long long &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<float             >(char *&, char *&, size_t &, size_t const, size_t const, float              &, std::vector<char> &, std::ifstream &, char const);
    template bool Read_Stream_Block_And_Parse_Number<double            >(char *&, char *&, size_t &, size_t const, size_t const, double             &, std::vector<char> &, std::ifstream &, char const);

    template <typename T>
    T Cin_Number(T const minimum_number_received,
                 T const maximum_number_received,
                 std::string const &ref_prefix_received)
    {
        T tmp_return(0);

        std::string tmp_string_digit;

        std::smatch tmp_smatch;

        std::regex tmp_regex("^([-+]?[0-9]+)$");

        do
        {
            PRINT(ref_prefix_received);

            getline(std::cin, tmp_string_digit);

            if(tmp_string_digit.empty()) { continue; }

            if(std::regex_match(tmp_string_digit,
                                tmp_smatch,
                                tmp_regex,
                                std::regex_constants::match_default))
            {
                try
                {
                    tmp_return = std::stoi(tmp_smatch[1u]);

                    if      constexpr (std::is_same<T, int               >::value) { tmp_return = std::stoi  (tmp_smatch[1u]); }
                    else if constexpr (std::is_same<T, long              >::value) { tmp_return = std::stol  (tmp_smatch[1u]); }
                    else if constexpr (std::is_same<T, long long         >::value) { tmp_return = std::stoll (tmp_smatch[1u]); }
                    else if constexpr (std::is_same<T, unsigned int      >::value
                                       ||
                                       std::is_same<T, unsigned long     >::value) { tmp_return = std::stoul (tmp_smatch[1u]); }
                    else if constexpr (std::is_same<T, unsigned long long>::value) { tmp_return = std::stoull(tmp_smatch[1u]); }
                    else { throw(std::logic_error("NotImplementedException")); }

                    if(     tmp_return < minimum_number_received) { continue; }
                    else if(tmp_return > maximum_number_received) { continue; }

                    return(tmp_return);
                }
                catch(...) { continue; }
            }
        } while(true);
    }

    template int                Cin_Number<int               >(int                const, int                const, std::string const &);
    template long               Cin_Number<long              >(long               const, long               const, std::string const &);
    template long long          Cin_Number<long long         >(long long          const, long long          const, std::string const &);
    template unsigned int       Cin_Number<unsigned int      >(unsigned int       const, unsigned int       const, std::string const &);
    template unsigned long      Cin_Number<unsigned long     >(unsigned long      const, unsigned long      const, std::string const &);
    template unsigned long long Cin_Number<unsigned long long>(unsigned long long const, unsigned long long const, std::string const &);

    template <typename T>
    T Cin_Number(T const minimum_number_received, std::string const &ref_prefix_received)
    {
        return(Cin_Number(minimum_number_received,
                          (std::numeric_limits<T>::max)(),
                          ref_prefix_received));
    }

    template int                Cin_Number<int               >(int                const, std::string const &);
    template long               Cin_Number<long              >(long               const, std::string const &);
    template long long          Cin_Number<long long         >(long long          const, std::string const &);
    template unsigned int       Cin_Number<unsigned int      >(unsigned int       const, std::string const &);
    template unsigned long      Cin_Number<unsigned long     >(unsigned long      const, std::string const &);
    template unsigned long long Cin_Number<unsigned long long>(unsigned long long const, std::string const &);

    template <typename T>
    T Cin_Real_Number(T const minimum_real_number_received,
                      T const maximum_real_number_received,
                      std::string const &ref_prefix_received)
    {
        T tmp_return(0);

        std::string tmp_string_digit;

        std::smatch tmp_smatch;

        std::regex tmp_regex("^([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$");

        do
        {
            PRINT(ref_prefix_received);

            getline(std::cin, tmp_string_digit);

            if(tmp_string_digit.empty()) { continue; }

            if(std::regex_match(tmp_string_digit,
                                tmp_smatch,
                                tmp_regex,
                                std::regex_constants::match_default))
            {
                try
                {
                    if      constexpr (std::is_same<T, float      >::value) { tmp_return = std::stof (tmp_smatch[1u]); }
                    else if constexpr (std::is_same<T, double     >::value) { tmp_return = std::stod (tmp_smatch[1u]); }
                    else if constexpr (std::is_same<T, double long>::value) { tmp_return = std::stold(tmp_smatch[1u]); }
                    else { throw(std::logic_error("NotImplementedException")); }

                    if(     tmp_return < minimum_real_number_received) { continue; }
                    else if(tmp_return > maximum_real_number_received) { continue; }

                    return(tmp_return);
                }
                catch(...) { continue; }
            }
        } while(true);
    }

    template float       Cin_Real_Number<float      >(float       const, float       const, std::string const &);
    template double      Cin_Real_Number<double     >(double      const, double      const, std::string const &);
    template double long Cin_Real_Number<double long>(double long const, double long const, std::string const &);

#if defined(COMPILE_ADEPT)
    #if defined(COMPILE_FLOAT)
        template <>
        adept::afloat Cin_Real_Number(adept::afloat const minimum_real_number_received,
                                      adept::afloat const maximum_real_number_received,
                                      std::string const &ref_prefix_received)
        {
            float tmp_return(0.0f);

            std::string tmp_string_digit;

            std::smatch tmp_smatch;

            std::regex tmp_regex("^([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$");

            do
            {
                PRINT(ref_prefix_received);

                getline(std::cin, tmp_string_digit);

                if(tmp_string_digit.empty()) { continue; }

                if(std::regex_match(tmp_string_digit,
                                    tmp_smatch,
                                    tmp_regex,
                                    std::regex_constants::match_default))
                {
                    try
                    {
                        tmp_return = std::stof(tmp_smatch[1u]);

                        if(     tmp_return < minimum_real_number_received) { continue; }
                        else if(tmp_return > maximum_real_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);
        }
    #elif defined(COMPILE_DOUBLE)
        template <>
        adept::adouble Cin_Real_Number(adept::adouble const minimum_real_number_received,
                                       adept::adouble const maximum_real_number_received,
                                       std::string const &ref_prefix_received)
        {
            double tmp_return(0.0);

            std::string tmp_string_digit;

            std::smatch tmp_smatch;

            std::regex tmp_regex("^([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$");

            do
            {
                PRINT(ref_prefix_received);

                getline(std::cin, tmp_string_digit);

                if(tmp_string_digit.empty()) { continue; }

                if(std::regex_match(tmp_string_digit,
                                    tmp_smatch,
                                    tmp_regex,
                                    std::regex_constants::match_default))
                {
                    try
                    {
                        tmp_return = std::stod(tmp_smatch[1u]);

                        if(     tmp_return < minimum_real_number_received) { continue; }
                        else if(tmp_return > maximum_real_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);
        }
    #endif // COMPILE_FLOAT || COMPILE_DOUBLE
#endif // COMPILE_ADEPT

    template <typename T>
    T Cin_Real_Number(T const minimum_real_number_received, std::string const &ref_prefix_received)
    {
        return(Cin_Real_Number(minimum_real_number_received,
                               (std::numeric_limits<T>::max)(),
                               ref_prefix_received));
    }

    template float       Cin_Real_Number<float      >(float       const, std::string const &);
    template double      Cin_Real_Number<double     >(double      const, std::string const &);
    template double long Cin_Real_Number<double long>(double long const, std::string const &);

#if defined(COMPILE_ADEPT)
    #if defined(COMPILE_FLOAT)
        template adept::afloat Cin_Real_Number<adept::afloat>(adept::afloat const, std::string const &);
    #elif defined(COMPILE_DOUBLE)
        template adept::adouble Cin_Real_Number<adept::adouble>(adept::adouble const, std::string const &);
    #endif // COMPILE_FLOAT || COMPILE_DOUBLE
#endif // COMPILE_ADEPT

    bool NoOrYes(std::string const &ref_prefix_received)
    {
        std::string tmp_string_option;

        bool tmp_is_digit;

        int tmp_option;

        while(true)
        {
            std::cout << ref_prefix_received << " (No:[0], Yes:[1]): ";

            getline(std::cin, tmp_string_option);

            if(tmp_string_option.empty() == false)
            {
                tmp_is_digit = true;

                for(char &tmp_ref_char : tmp_string_option)
                {
                    if(isdigit(tmp_ref_char) == false)
                    {
                        tmp_is_digit = false;

                        break;
                    }
                }

                if(tmp_is_digit)
                {
                    tmp_option = std::stoi(tmp_string_option);

                    if(tmp_string_option.size() == 1_zu)
                    {
                        if(     tmp_option == 1) { return(true ); }
                        else if(tmp_option == 0) { return(false); }
                    }
                }
                else if(tmp_string_option.size() <= 3_zu)
                {
                    std::transform(tmp_string_option.begin(),
                                   tmp_string_option.end(),
                                   tmp_string_option.begin(),
                                   ::tolower);
                    if(strcmp(tmp_string_option.c_str(), "yes") == 0
                       ||
                       strcmp(tmp_string_option.c_str(), "ye" ) == 0
                       ||
                       strcmp(tmp_string_option.c_str(), "y"  ) == 0)
                    { return(true); }
                    else if(strcmp(tmp_string_option.c_str(), "no") == 0
                            ||
                            strcmp(tmp_string_option.c_str(), "n" ) == 0)
                    { return(false); }
                }
            }
        }
    }

    std::string Get__Time(std::string format_received, bool const use_local_time_received)
    {
        std::ostringstream tmp_ostringstream;

        if(format_received.empty())
        {
            if(use_local_time_received) { format_received = "[L:%d/%m/%Y %Hh:%Mm:%Ss]" ; }
            else                        { format_received = "[GM:%d/%m/%Y %Hh:%Mm:%Ss]"; }
        }

        time_t tmp_time_t(std::time(nullptr));

    #if defined(COMPILE_WINDOWS)
        struct tm tmp_tm;

        if(use_local_time_received) { localtime_s(&tmp_tm, &tmp_time_t); }
        else                        { gmtime_s(&tmp_tm, &tmp_time_t)   ; }

        tmp_ostringstream << std::put_time(&tmp_tm, format_received.c_str());
    #elif defined(COMPILE_LINUX)
        struct tm *tmp_ptr_tm;

        if(use_local_time_received) { tmp_ptr_tm = localtime(&tmp_time_t); }
        else                        { tmp_ptr_tm = gmtime(&tmp_time_t)   ; }

        tmp_ostringstream << std::put_time(tmp_ptr_tm, format_received.c_str());
    #endif // COMPILE_WINDOWS || COMPILE_LINUX

        return(tmp_ostringstream.str());
    }

    std::string Get__Time_Elapse(double const time_elapse_received)
    {
        std::string tmp_string;

        if(     time_elapse_received <= 0.000'000'999) { tmp_string = std::to_string(time_elapse_received * 1e+9) + "ns"; } // nanoseconds
        else if(time_elapse_received <= 0.000'999    ) { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received * 1e+6, 3u) + "us"; } // microseconds μs
        else if(time_elapse_received <= 0.999        ) { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received * 1e+3, 3u) + "ms"; } // milliseconds
        else if(time_elapse_received <= 59.0         ) { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received, 3u) + "s"; } // seconds
        else if(time_elapse_received <= 3599.0       )
        {
            tmp_string =  std::to_string(static_cast<unsigned int>(floor(time_elapse_received / 60.0))) + "m:"; // minute
            tmp_string += std::to_string(static_cast<unsigned int>(time_elapse_received) % 60u) + "s:"; // seconde
            tmp_string += To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
        }
        else if(time_elapse_received <= 86'399.0)
        {
            double const tmp_minutes(static_cast<double>(static_cast<unsigned int>(time_elapse_received) % 3600u) / 60.0);

            tmp_string =  std::to_string(static_cast<unsigned int>(floor(time_elapse_received / 3600.0))) + "h:"; // hour
            tmp_string += std::to_string(static_cast<unsigned int>(floor(tmp_minutes))) + "m:"; // minute
            tmp_string += std::to_string(static_cast<unsigned int>(tmp_minutes) % 60u) + "s:"; // seconde
            tmp_string += To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
        }
        else { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received, 3u) + "s"; } // seconde

        return(tmp_string);
    }

    std::string To_Upper(std::string string_to_uppercase_received)
    {
        std::locale tmp_locale;

        for(std::string::size_type i(0); i != string_to_uppercase_received.length(); ++i)
        {
            string_to_uppercase_received[i] = std::toupper(string_to_uppercase_received[i], tmp_locale);
        }

        return(string_to_uppercase_received);
    }

    void Print(char const *ptr_fmt_received, ...)
    {
        std::cout << MyEA::Time::Get__DateTimeFull() << ": ";

        va_list args; va_start(args, ptr_fmt_received);

        vprintf(ptr_fmt_received, args);

        va_end(args);

        std::cout << std::endl;
    }

    void Print_With_Prefix(std::string const &ref_prefix_received, char const *ptr_fmt_received, ...)
    {
        std::cout << MyEA::Time::Get__DateTimeFull() << ": " << ref_prefix_received;

        va_list args; va_start(args, ptr_fmt_received);

        vprintf(ptr_fmt_received, args);

        va_end(args);

        std::cout << std::endl;
    }

    void Find_And_Replace(std::string       &ref_source,
                          std::string const &ref_find,
                          std::string const &ref_replace)
    {
        for(std::string::size_type i(0); (i = ref_source.find(ref_find, i)) != std::string::npos;)
        {
            ref_source.replace(i,
                               ref_find.length(),
                               ref_replace);

            i += ref_replace.length();
        }
    }

    bool Regex_Read_Input(int &ref_result_received,
                          std::string const &ref_line_received,
                          std::regex const &ref_regex_received)
    {
        std::smatch tmp_smatch;

        if(std::regex_match(ref_line_received,
                            tmp_smatch,
                            ref_regex_received,
                            std::regex_constants::match_default) == false)
        {
            MyEA::String::Error("Can not read this line correctly: %s.", ref_line_received.c_str());

            return(false);
        }
        else if(tmp_smatch.size() > 2_zu)
        {
            MyEA::String::Error("More than one result find at line: %s.", ref_line_received.c_str());

            return(false);
        }
        else { ref_result_received = std::stoi(tmp_smatch[1u]); }

        return(true);
    }

    bool Regex_Read_Input(float &ref_result_received,
                          std::string const &ref_line_received,
                          std::regex const &ref_regex_received)
    {
        std::smatch tmp_smatch;

        if(std::regex_match(ref_line_received,
                            tmp_smatch,
                            ref_regex_received,
                            std::regex_constants::match_default) == false)
        {
            MyEA::String::Error("Can not read this line correctly: %s.", ref_line_received.c_str());

            return(false);
        }
        else if(tmp_smatch.size() > 2_zu)
        {
            MyEA::String::Error("More than one result find at line: %s.", ref_line_received.c_str());

            return(false);
        }
        else { ref_result_received = std::stof(tmp_smatch[1u]); }

        return(true);
    }

    std::string Execute_Command(char const *const tmp_ptr_command_received)
    {
    #if defined(COMPILE_WINDOWS)
        FILE *tmp_ptr_file_command(_popen(tmp_ptr_command_received, "r"));
    #elif defined(COMPILE_LINUX)
        FILE *tmp_ptr_file_command(popen(tmp_ptr_command_received, "r"));
    #endif // COMPILE_WINDOWS || COMPILE_LINUX

        if(tmp_ptr_file_command == NULL)
        {
            MyEA::String::Error("An error has been triggered from the `popen(%s)` function.", tmp_ptr_command_received);

            return("");
        }

        char tmp_buffer[1024u];

        std::string tmp_output("");

        while(fgets(tmp_buffer, sizeof(tmp_buffer), tmp_ptr_file_command) != NULL) { tmp_output += tmp_buffer; }

        if(ferror(tmp_ptr_file_command) != 0)
        {
            MyEA::String::Error("An error has been triggered from the `fgets(%s, %zu, %s)` function.",
                                tmp_buffer,
                                sizeof(tmp_buffer),
                                tmp_ptr_command_received);

            return(tmp_output);
        }

    #if defined(COMPILE_WINDOWS)
        if(_pclose(tmp_ptr_file_command) == -1)
    #elif defined(COMPILE_LINUX)
        if(pclose(tmp_ptr_file_command) == -1)
    #endif // COMPILE_WINDOWS || COMPILE_LINUX
        {
            MyEA::String::Error("An error has been triggered from the `pclose(%s)` function.", tmp_ptr_command_received);

            return(tmp_output);
        }

        return(tmp_output);
    }
}
