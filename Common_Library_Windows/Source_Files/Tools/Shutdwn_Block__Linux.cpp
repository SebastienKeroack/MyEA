#include "stdafx.hpp"

#include <Tools/Shutdown_Block__Linux.hpp>
#include <Tools/Configuration.hpp>
#include <Strings/String.hpp>
#include <Tools/Reallocate.hpp>

class Shutdown_Block *ptr_global_Shutdown_Block = nullptr;

Shutdown_Block::Shutdown_Block(std::string const &ref_window_name_received) : _window_name(ref_window_name_received)
{
    if(this->Check_Systemd_Version() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Systemd_Version()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        this->Query_Shutdown();
    }
}

bool Shutdown_Block::Check_Systemd_Version(void) const
{
    std::string tmp_output_systemd_version(MyEA::String::Execute_Command("systemd --version"));

    if(tmp_output_systemd_version.empty())
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Execute_Command(\"busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager PreparingForSleep\")\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    std::string::size_type const tmp_character_position(tmp_output_systemd_version.find_first_of(" "));

    if(tmp_character_position == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"find_first_of(\" \")\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
    }
        
    // Substring "systemd ".
    tmp_output_systemd_version = tmp_output_systemd_version.substr(tmp_character_position + 1);

    int tmp_systemd_version(0);
        
    try { tmp_systemd_version = std::stoi(tmp_output_systemd_version); }
    catch(std::invalid_argument&)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"stoi(%s)\" function. return invalid argument. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_output_systemd_version.c_str(),
                                 __LINE__);

        return(false);
    }
    catch(std::out_of_range&)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"stoi(%s)\" function. return out of range. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_output_systemd_version.c_str(),
                                 __LINE__);

        return(false);
    }
    catch(...)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"stoi(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_output_systemd_version.c_str(),
                                 __LINE__);

        return(false);
    }

    if(tmp_systemd_version < 220)
    {
        PRINT_FORMAT("%s: %s: ERROR: Systemd current version %d need to be update to the version 220 or greater. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_systemd_version,
                                 __LINE__);

        return(false);
    }

    return(true);
}

void Shutdown_Block::Query_Shutdown(void)
{
    this->_on_shutdown = true;
        
    for(unsigned int tmp_boolean_index(0u); tmp_boolean_index != this->_number_boolean; ++tmp_boolean_index)
    { this->_ptr_array_ptr_shutdown_boolean[tmp_boolean_index]->store(true); }
}

void Shutdown_Block::Deallocate__Array_Shutdown_Boolean(void) { SAFE_DELETE_ARRAY(this->_ptr_array_ptr_shutdown_boolean); }

void Shutdown_Block::Initialize_Static_Shutdown_Block(void)
{ ptr_global_Shutdown_Block = this; }
    
bool Shutdown_Block::Get__On_Shutdown(void) const { return(this->_on_shutdown); }
    
bool Shutdown_Block::Create_Shutdown_Block(bool const use_ctrl_handler_received)
{
    if(this->_initialize == false)
    {
        this->Initialize_Static_Shutdown_Block();

        int tmp_return_code;

        // Connect to the system bus.
        if((tmp_return_code = sd_bus_open_system(&this->_ptr_sd_bus)) < 0)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"sd_bus_open_system()\" function. Code: %d. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_return_code,
                                     __LINE__);

            sd_bus_error_free(&this->_sd_bus_error);

            return(false);
        }

        // Issue the method call and store the handle.
        if((tmp_return_code = sd_bus_call_method(this->_ptr_sd_bus,
                                                                      "org.freedesktop.login1",                 // Service to contact.
                                                                      "/org/freedesktop/login1",                // Object path.
                                                                      "org.freedesktop.login1.Manager",    // Interface name.
                                                                      "Inhibit",                                         // Method name.
                                                                      &this->_sd_bus_error,                     // Object to return error in.
                                                                      &this->_ptr_sd_bus_handle,             // Return message on success.
                                                                      "ssss",                                           // Input signature.
                                                                      "shutdown",                                    // argument: what.
                                                                      this->_window_name.c_str(),            // argument: who.
                                                                      "Closing in progress...",                   // argument: why.
                                                                      "delay")) < 0)                                  // argument: mode.
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"sd_bus_call_method()\" function. Code: %d | %s. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_return_code,
                                     this->_sd_bus_error.message,
                                     __LINE__);

            sd_bus_unref(this->_ptr_sd_bus);
                
            sd_bus_error_free(&this->_sd_bus_error);

            return(false);
        }

        this->_initialize = true;

        return(true);
    }
    else { return(false); }
}

bool Shutdown_Block::Remove_Shutdown_Block(void)
{
    if(this->_initialize)
    {
        if(this->_asynchronous_mode)
        {
            this->_asynchronous_mode = false;

            if(this->_asynchronous_thread.joinable()) { this->_asynchronous_thread.join(); }
        }

        sd_bus_message_unref(this->_ptr_sd_bus_handle);

        this->_initialize = false;

        return(true);
    }
    else { return(false); }
}

bool Shutdown_Block::Peak_Message(void)
{
    if(this->_initialize && this->_on_shutdown == false)
    {
        int tmp_return_code,
            tmp_preparing_for_shutdown;

        // Issue the method call and store the value in a boolean.
        if((tmp_return_code = sd_bus_get_property(this->_ptr_sd_bus,
                                                                       "org.freedesktop.login1",                       // Service to contact.
                                                                       "/org/freedesktop/login1",                      // Object path.
                                                                       "org.freedesktop.login1.Manager",          // Interface name.
                                                                       "PreparingForShutdown",                       // Method name.
                                                                       &this->_sd_bus_error,                           // Object to return error in.
                                                                       &this->_ptr_sd_bus_message,               // Reply.
                                                                       "b"                                                       // Input signature.
                                                                       )) < 0)       // Output.
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"sd_bus_get_property()\" function. Code: %d | %s. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_return_code,
                                     this->_sd_bus_error.message,
                                     __LINE__);

            sd_bus_error_free(&this->_sd_bus_error);

            return(false);
        }
        
        // Read.
        if((tmp_return_code = sd_bus_message_read(this->_ptr_sd_bus_message, "b", &tmp_preparing_for_shutdown)) < 0)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"sd_bus_message_read()\" function. Code: %d. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_return_code,
                                     __LINE__);

            return(false);
        }
        
        if(tmp_preparing_for_shutdown != 0) { this->Query_Shutdown(); }
    }

    return(true);
}
    
bool Shutdown_Block::Peak_Message_Async(void)
{
    if(this->_initialize == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Shutdown block not initialized. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_asynchronous_mode)
    {
        PRINT_FORMAT("%s: %s: ERROR: Asynchronous mode is already enabled. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->_on_shutdown == false)
    {
        this->_asynchronous_mode = true;

        this->_asynchronous_thread = std::thread(&Shutdown_Block::_Peak_Message_Async, this);
    }

    return(true);
}
    
bool Shutdown_Block::_Peak_Message_Async(void)
{
    if(this->_initialize)
    {
        if(this->Peak_Message() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Peak_Message()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        while(this->_on_shutdown == false && this->_asynchronous_mode)
        {
            std::this_thread::sleep_for(std::chrono::seconds(3));

            if(this->Peak_Message() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Peak_Message()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
    }

    return(true);
}
    
bool Shutdown_Block::Push_Back(std::atomic<bool> *const ptr_shutdown_boolean)
{
    if(this->_ptr_array_ptr_shutdown_boolean == nullptr)
    {
        std::atomic<bool> **tmp_ptr_array_ptr_shutdown_boolean = new std::atomic<bool>*[1u];

        if(tmp_ptr_array_ptr_shutdown_boolean == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(std::atomic<bool>*),
                                     __LINE__);

            return(false);
        }

        this->_ptr_array_ptr_shutdown_boolean = tmp_ptr_array_ptr_shutdown_boolean;

        this->_ptr_array_ptr_shutdown_boolean[0u] = ptr_shutdown_boolean;

        ++this->_number_boolean;
    }
    else
    {
        this->_ptr_array_ptr_shutdown_boolean = Memory::reallocate_pointers_array_cpp<std::atomic<bool>*>(this->_ptr_array_ptr_shutdown_boolean,
                                                                                                                                                              this->_number_boolean + 1_zu,
                                                                                                                                                              this->_number_boolean,
                                                                                                                                                              true);

        if(this->_ptr_array_ptr_shutdown_boolean == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     (this->_number_boolean + 1_zu) * sizeof(std::atomic<bool>*),
                                     __LINE__);

            return(false);
        }

        this->_ptr_array_ptr_shutdown_boolean[this->_number_boolean] = ptr_shutdown_boolean;

        ++this->_number_boolean;
    }

    return(true);
}

Shutdown_Block::~Shutdown_Block(void)
{
    this->Remove_Shutdown_Block();
        
    if(this->_ptr_sd_bus != NULL) { sd_bus_unref(this->_ptr_sd_bus); }

    sd_bus_error_free(&this->_sd_bus_error);

    this->Deallocate__Array_Shutdown_Boolean();
}