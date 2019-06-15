#include "stdafx.hpp"

#if defined(COMPILE_LINUX)
    #include <cstring>
    #include <ctime>
#endif // COMPILE_LINUX

#include <Strings/String.hpp>

#include <boost/spirit/home/x3.hpp>

#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <regex>

namespace MyEA
{
    namespace String
    {
        template <typename T>
        bool Parse_Number(char *&ptr_array_characters_received,
                                      char *const ptr_last_character_received,
                                      T &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::char_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template bool Parse_Number<char>(char *&, char *const, char &);
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        short &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::short_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        int &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::int_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        long &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::long_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        long long &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::long_long[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        unsigned short &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::ushort_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        unsigned int &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::uint_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        unsigned long &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::ulong_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <> bool Parse_Number(char *&ptr_array_characters_received,
                                                        char *const ptr_last_character_received,
                                                        unsigned long long &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::ulong_long[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template <typename T>
        bool Parse_Real_Number(char *&ptr_array_characters_received,
                                              char *const ptr_last_character_received,
                                              T &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::float_[tmp_Assign],
                                                                  boost::spirit::x3::ascii::space));
        }
        
        template bool Parse_Real_Number<float>(char *&, char *const, float &);
        
        template <> bool Parse_Real_Number(char *&ptr_array_characters_received,
                                                                char *const ptr_last_character_received,
                                                                double &ref_output_received)
        {
            auto tmp_Assign([&ref_output_received](auto &ref_ctx_received) { ref_output_received = _attr(ref_ctx_received); });
    
            return(boost::spirit::x3::phrase_parse(ptr_array_characters_received,
                                                                  ptr_last_character_received,
                                                                  boost::spirit::x3::double_[tmp_Assign],
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

            // Remaining characters based on current position.
            ref_ifstream_received.seekg(0, std::ios::end);

            if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not sets the position of the next character to be extracted from the input stream. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            size_t const tmp_remaining_size(static_cast<size_t>(ref_ifstream_received.tellg() - tmp_current_tellg)),
                               tmp_block_size(tmp_remaining_size < desired_block_size_received ? tmp_remaining_size : desired_block_size_received);
            
            if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not gets the position of the next character to be extracted from the input stream. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            // Return to the last position.
            ref_ifstream_received.seekg(tmp_current_tellg, std::ios::beg);
            
            if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not sets the position (%zu | beg) of the next character to be extracted from the input stream. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<size_t>(tmp_current_tellg),
                                         __LINE__);

                return(false);
            }

            // If not enought space in the buffer resize it.
            if(ref_vector_buffers_received.size() < tmp_block_size) { ref_vector_buffers_received.resize(tmp_block_size); }

            // Read block into buffers.
            ref_ifstream_received.read(&ref_vector_buffers_received[0], tmp_block_size);
            
            if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly the file. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

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
                    // If not enought space in the buffer resize it.
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
                    PRINT_FORMAT("%s: %s: ERROR: Can not read properly the character (%c). At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             ref_vector_buffers_received[ref_block_size_received - 1],
                                             __LINE__);

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
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block(%zu, %zu, %zu, vector, ifstream, '\\n')\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             ref_block_size_received,
                                             desired_block_size_received,
                                             step_block_size_received,
                                             __LINE__);

                    return(false);
                }

                ptr_array_characters_received = &ref_vector_buffers_received[0];
                ptr_last_character_received = ptr_array_characters_received + ref_block_size_received;
            }
                    
            if(MyEA::String::Parse_Number<T>(ptr_array_characters_received,
                                                                ptr_last_character_received,
                                                                ref_output_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Parse_Number()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            return(true);
        }
        
        template bool Read_Stream_Block_And_Parse_Number<char>(char *&, char *&, size_t &, size_t const, size_t const, char &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<int>(char *&, char *&, size_t &, size_t const, size_t const, int &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<short>(char *&, char *&, size_t &, size_t const, size_t const, short &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<long>(char *&, char *&, size_t &, size_t const, size_t const, long &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<long long>(char *&, char *&, size_t &, size_t const, size_t const, long long &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<unsigned short>(char *&, char *&, size_t &, size_t const, size_t const, unsigned short &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<unsigned int>(char *&, char *&, size_t &, size_t const, size_t const, unsigned int &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<unsigned long>(char *&, char *&, size_t &, size_t const, size_t const, unsigned long &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Number<unsigned long long>(char *&, char *&, size_t &, size_t const, size_t const, unsigned long long &, std::vector<char> &, std::ifstream &, char const);
        
        template <typename T>
        bool Read_Stream_Block_And_Parse_Real_Number(char *&ptr_array_characters_received,
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
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block(%zu, %zu, %zu, vector, ifstream, '\\n')\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             ref_block_size_received,
                                             desired_block_size_received,
                                             step_block_size_received,
                                             __LINE__);

                    return(false);
                }

                ptr_array_characters_received = &ref_vector_buffers_received[0];
                ptr_last_character_received = ptr_array_characters_received + ref_block_size_received;
            }
                    
            if(MyEA::String::Parse_Real_Number<T>(ptr_array_characters_received,
                                                                        ptr_last_character_received,
                                                                        ref_output_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Parse_Real_Number()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            return(true);
        }
        
        template bool Read_Stream_Block_And_Parse_Real_Number<float>(char *&, char *&, size_t &, size_t const, size_t const, float &, std::vector<char> &, std::ifstream &, char const);
        template bool Read_Stream_Block_And_Parse_Real_Number<double>(char *&, char *&, size_t &, size_t const, size_t const, double &, std::vector<char> &, std::ifstream &, char const);
        
        template <typename T>
        T Cin_Number(T const minimum_number_received, std::string const &ref_prefix_received)
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

                        if(tmp_return < minimum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template int Cin_Number<int>(int const minimum_number_received, std::string const &ref_prefix_received);
        
        template <> long Cin_Number(long const minimum_number_received, std::string const &ref_prefix_received)
        {
            long tmp_return(0l);

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
                        tmp_return = std::stol(tmp_smatch[1u]);

                        if(tmp_return < minimum_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> long long Cin_Number(long long const minimum_number_received, std::string const &ref_prefix_received)
        {
            long long tmp_return(0ll);

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
                        tmp_return = std::stoll(tmp_smatch[1u]);

                        if(tmp_return < minimum_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> unsigned int Cin_Number(unsigned int const minimum_number_received, std::string const &ref_prefix_received)
        {
            unsigned int tmp_return(0u);

            std::string tmp_string_digit;
    
            std::smatch tmp_smatch;
            
            std::regex tmp_regex("^(\\+?[0-9]+)$");

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
                        tmp_return = static_cast<unsigned int>(std::stoul(tmp_smatch[1u]));

                        if(tmp_return < minimum_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> unsigned long Cin_Number(unsigned long const minimum_number_received, std::string const &ref_prefix_received)
        {
            unsigned long tmp_return(0ul);

            std::string tmp_string_digit;
    
            std::smatch tmp_smatch;
            
            std::regex tmp_regex("^(\\+?[0-9]+)$");

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
                        tmp_return = std::stoul(tmp_smatch[1u]);

                        if(tmp_return < minimum_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> unsigned long long Cin_Number(unsigned long long const minimum_number_received, std::string const &ref_prefix_received)
        {
            unsigned long long tmp_return(0ull);

            std::string tmp_string_digit;
    
            std::smatch tmp_smatch;
            
            std::regex tmp_regex("^(\\+?[0-9]+)$");

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
                        tmp_return = std::stoull(tmp_smatch[1u]);

                        if(tmp_return < minimum_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
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

                        if(tmp_return < minimum_number_received) { continue; }
                        else if(tmp_return > maximum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template int Cin_Number<int>(int const, int const, std::string const &);
        
        template <> long Cin_Number(long const minimum_number_received,
                                                    long const maximum_number_received,
                                                    std::string const &ref_prefix_received)
        {
            long tmp_return(0l);

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
                        tmp_return = std::stol(tmp_smatch[1u]);
                        
                        if(tmp_return < minimum_number_received) { continue; }
                        else if(tmp_return > maximum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> long long Cin_Number(long long const minimum_number_received,
                                                           long long const maximum_number_received,
                                                           std::string const &ref_prefix_received)
        {
            long long tmp_return(0ll);

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
                        tmp_return = std::stoll(tmp_smatch[1u]);
                        
                        if(tmp_return < minimum_number_received) { continue; }
                        else if(tmp_return > maximum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> unsigned int Cin_Number(unsigned int const minimum_number_received,
                                                                unsigned int const maximum_number_received,
                                                                std::string const &ref_prefix_received)
        {
            unsigned int tmp_return(0u);

            std::string tmp_string_digit;
    
            std::smatch tmp_smatch;
            
            std::regex tmp_regex("^(\\+?[0-9]+)$");

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
                        tmp_return = static_cast<unsigned int>(std::stoul(tmp_smatch[1u]));
                        
                        if(tmp_return < minimum_number_received) { continue; }
                        else if(tmp_return > maximum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> unsigned long Cin_Number(unsigned long const minimum_number_received,
                                                                  unsigned long const maximum_number_received,
                                                                  std::string const &ref_prefix_received)
        {
            unsigned long tmp_return(0ul);

            std::string tmp_string_digit;
    
            std::smatch tmp_smatch;
            
            std::regex tmp_regex("^(\\+?[0-9]+)$");

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
                        tmp_return = std::stoul(tmp_smatch[1u]);
                        
                        if(tmp_return < minimum_number_received) { continue; }
                        else if(tmp_return > maximum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template <> unsigned long long Cin_Number(unsigned long long const minimum_number_received,
                                                                         unsigned long long const maximum_number_received,
                                                                         std::string const &ref_prefix_received)
        {
            unsigned long long tmp_return(0ull);

            std::string tmp_string_digit;
    
            std::smatch tmp_smatch;
            
            std::regex tmp_regex("^(\\+?[0-9]+)$");

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
                        tmp_return = std::stoull(tmp_smatch[1u]);
                        
                        if(tmp_return < minimum_number_received) { continue; }
                        else if(tmp_return > maximum_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }

        template <typename T>
        T Cin_Real_Number(T const minimum_real_number_received, std::string const &ref_prefix_received)
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
                        tmp_return = std::stof(tmp_smatch[1u]);

                        if(tmp_return < minimum_real_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }

        template float Cin_Real_Number<float>(float const minimum_real_number_received, std::string const &ref_prefix_received);

        template <> double Cin_Real_Number(double const minimum_real_number_received, std::string const &ref_prefix_received)
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

                        if(tmp_return < minimum_real_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }

        template <> long double Cin_Real_Number(long double const minimum_real_number_received, std::string const &ref_prefix_received)
        {
            long double tmp_return(0.0l);

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
                        tmp_return = std::stold(tmp_smatch[1u]);

                        if(tmp_return < minimum_real_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
    #if defined(COMPILE_ADEPT)
        #if defined(COMPILE_FLOAT)
            template <> adept::afloat Cin_Real_Number<adept::afloat>(adept::afloat const minimum_real_number_received, std::string const &ref_prefix_received)
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

                            if(tmp_return < minimum_real_number_received) { continue; }
                        
                            return(tmp_return);
                        }
                        catch(...) { continue; }
                    }
                } while(true);

                return(adept::afloat(tmp_return));
            }
        #elif defined(COMPILE_DOUBLE)
            template <> adept::adouble Cin_Real_Number<adept::adouble>(adept::adouble const minimum_real_number_received, std::string const &ref_prefix_received)
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

                            if(tmp_return < minimum_real_number_received) { continue; }
                        
                            return(tmp_return);
                        }
                        catch(...) { continue; }
                    }
                } while(true);

                return(adept::adouble(tmp_return));
            }
        #endif // COMPILE_FLOAT || COMPILE_DOUBLE
    #endif // COMPILE_ADEPT

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
                        tmp_return = std::stof(tmp_smatch[1u]);

                        if(tmp_return < minimum_real_number_received) { continue; }
                        else if(tmp_return > maximum_real_number_received) { continue; }

                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
        template float Cin_Real_Number<float>(float const, float const, std::string const &);

        template <> double Cin_Real_Number(double const minimum_real_number_received,
                                                                double const maximum_real_number_received,
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

                        if(tmp_return < minimum_real_number_received) { continue; }
                        else if(tmp_return > maximum_real_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }

        template <> long double Cin_Real_Number(long double const minimum_real_number_received,
                                                                       long double const maximum_real_number_received,
                                                                       std::string const &ref_prefix_received)
        {
            long double tmp_return(0.0l);

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
                        tmp_return = std::stold(tmp_smatch[1u]);

                        if(tmp_return < minimum_real_number_received) { continue; }
                        else if(tmp_return > maximum_real_number_received) { continue; }
                        
                        return(tmp_return);
                    }
                    catch(...) { continue; }
                }
            } while(true);

            return(tmp_return);
        }
        
    #if defined(COMPILE_ADEPT)
        #if defined(COMPILE_FLOAT)
            template <> adept::afloat Cin_Real_Number(adept::afloat const minimum_real_number_received,
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

                            if(tmp_return < minimum_real_number_received) { continue; }
                            else if(tmp_return > maximum_real_number_received) { continue; }

                            return(tmp_return);
                        }
                        catch(...) { continue; }
                    }
                } while(true);

                return(adept::afloat(tmp_return));
            }
        #elif defined(COMPILE_DOUBLE)
            template <> adept::adouble Cin_Real_Number(adept::adouble const minimum_real_number_received,
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

                            if(tmp_return < minimum_real_number_received) { continue; }
                            else if(tmp_return > maximum_real_number_received) { continue; }

                            return(tmp_return);
                        }
                        catch(...) { continue; }
                    }
                } while(true);

                return(adept::adouble(tmp_return));
            }
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

                        if(tmp_string_option.size() == 1u)
                        {
                            if(tmp_option == 1) { return(true); }
                            else if(tmp_option == 0) { return(false); }
                        }
                    }
                    else if(tmp_string_option.size() <= 3u)
                    {
                        std::transform(tmp_string_option.begin(),
                                               tmp_string_option.end(),
                                               tmp_string_option.begin(),
                                               ::tolower);
                        if(strcmp(tmp_string_option.c_str(), "yes") == 0
                           ||
                           strcmp(tmp_string_option.c_str(), "ye") == 0
                           ||
                           strcmp(tmp_string_option.c_str(), "y") == 0
                           ||
                           strcmp(tmp_string_option.c_str(), "oui") == 0
                           ||
                           strcmp(tmp_string_option.c_str(), "ou") == 0
                           ||
                           strcmp(tmp_string_option.c_str(), "o") == 0)
                        { return(true); }
                        else if(strcmp(tmp_string_option.c_str(), "non") == 0
                                   ||
                                   strcmp(tmp_string_option.c_str(), "no") == 0
                                   ||
                                   strcmp(tmp_string_option.c_str(), "n") == 0)
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
                if(use_local_time_received) { format_received = "[L:%d/%m/%Y %Hh:%Mm:%Ss]"; }
                else { format_received = "[GM:%d/%m/%Y %Hh:%Mm:%Ss]"; }
            }

            time_t tmp_time_t(std::time(nullptr));

        #if defined(COMPILE_WINDOWS)
            struct tm tmp_tm;

            if(use_local_time_received) { localtime_s(&tmp_tm, &tmp_time_t); }
            else { gmtime_s(&tmp_tm, &tmp_time_t); }

            tmp_ostringstream << std::put_time(&tmp_tm, format_received.c_str());
        #elif defined(COMPILE_LINUX)
            struct tm *tmp_ptr_tm;
            
            if(use_local_time_received) { tmp_ptr_tm = localtime(&tmp_time_t); }
            else { tmp_ptr_tm = gmtime(&tmp_time_t); }

            tmp_ostringstream << std::put_time(tmp_ptr_tm, format_received.c_str());
        #endif // COMPILE_WINDOWS || COMPILE_LINUX

            return(tmp_ostringstream.str());
        }

        std::string Get__Time_Elapse(double const time_elapse_received)
        {
            std::string tmp_string;

            if(time_elapse_received <= 0.000'000'999) { tmp_string = std::to_string(time_elapse_received * 1e+9) + "ns"; } // nanoseconds
            else if(time_elapse_received <= 0.000'999) { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received * 1e+6, 3u) + "us"; } // microseconds μs
            else if(time_elapse_received <= 0.999) { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received * 1e+3, 3u) + "ms"; } // milliseconds
            else if(time_elapse_received <= 59.0) { tmp_string = To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received, 3u) + "s"; } // seconds
            else if(time_elapse_received <= 3599.0)
            {
                tmp_string = std::to_string(static_cast<unsigned int>(floor(time_elapse_received / 60.0))) + "m:"; // minute
                tmp_string += std::to_string(static_cast<unsigned int>(time_elapse_received) % 60u) + "s:"; // seconde
                tmp_string += To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
            }
            else if(time_elapse_received <= 86'399.0)
            {
                double const tmp_minutes(static_cast<double>(static_cast<unsigned int>(time_elapse_received) % 3600u) / 60.0);

                tmp_string = std::to_string(static_cast<unsigned int>(floor(time_elapse_received / 3600.0))) + "h:"; // hour
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
            { string_to_uppercase_received[i] = std::toupper(string_to_uppercase_received[i], tmp_locale); }

            return(string_to_uppercase_received);
        }
        
        void Find_And_Replace(std::string &ref_source,std::string const &ref_find, std::string const &ref_replace)
        {
            for(std::string::size_type i = 0; (i = ref_source.find(ref_find, i)) != std::string::npos;)
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
            std::smatch tmp_regex_smatch;

            if(std::regex_match(ref_line_received,
                                         tmp_regex_smatch,
                                         ref_regex_received,
                                         std::regex_constants::match_default) == false)
            {
                PRINT_FORMAT("%s: ERROR: Can not read this line correctly: %s" NEW_LINE,
                                         __FUNCTION__,
                                         ref_line_received.c_str());

                return(false);
            }
            else if(tmp_regex_smatch.size() > 2u)
            {
                PRINT_FORMAT("%s: ERROR: More than one result find at line: %s" NEW_LINE,
                                         __FUNCTION__,
                                         ref_line_received.c_str());

                return(false);
            }
            else
            { ref_result_received = std::stoi(tmp_regex_smatch[1u]); }

            return(true);
        }

        bool Regex_Read_Input(float &ref_result_received,
                                           std::string const &ref_line_received,
                                           std::regex const &ref_regex_received)
        {
            std::smatch tmp_regex_smatch;

            if(std::regex_match(ref_line_received,
                                         tmp_regex_smatch,
                                         ref_regex_received,
                                         std::regex_constants::match_default) == false)
            {
                PRINT_FORMAT("%s: ERROR: Can not read this line correctly: %s" NEW_LINE,
                                         __FUNCTION__,
                                         ref_line_received.c_str());

                return(false);
            }
            else if(tmp_regex_smatch.size() > 2u)
            {
                PRINT_FORMAT("%s: ERROR: More than one result find at line: %s" NEW_LINE,
                                         __FUNCTION__,
                                         ref_line_received.c_str());

                return(false);
            }
            else
            { ref_result_received = std::stof(tmp_regex_smatch[1u]); }

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
                printf("%s: ERROR: From \"popen(%s, r)\"." NEW_LINE,
                        __FUNCTION__,
                        tmp_ptr_command_received);

                return("");
            }

            char tmp_buffer[1024u];

            std::string tmp_output("");

            while(fgets(tmp_buffer, sizeof(tmp_buffer), tmp_ptr_file_command) != NULL)
            { tmp_output += tmp_buffer; }

            if(ferror(tmp_ptr_file_command) != 0)
            {
                printf("%s: ERROR: From \"fgets()\"." NEW_LINE, __FUNCTION__);

                return(tmp_output);
            }
            
        #if defined(COMPILE_WINDOWS)
            if(_pclose(tmp_ptr_file_command) == -1)
        #elif defined(COMPILE_LINUX)
            if(pclose(tmp_ptr_file_command) == -1)
        #else // COMPILE_WINDOWS || COMPILE_LINUX
            if(false)
        #endif // COMPILE_WINDOWS || COMPILE_LINUX
            {
                printf("%s: ERROR: From \"pclose()\"." NEW_LINE, __FUNCTION__);

                return(tmp_output);
            }

            return(tmp_output);
        }

    }
}
