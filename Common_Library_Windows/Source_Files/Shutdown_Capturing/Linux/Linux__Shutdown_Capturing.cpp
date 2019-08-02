#include "stdafx.hpp"

#include <Shutdown_Capturing/Linux/Shutdown_Capturing.hpp>
#include <Configuration/Configuration.hpp>
#include <Strings/String.hpp>
#include <Reallocate/Reallocate.hpp>

class Shutdown_Capturing *ptr_global_Shutdown_Block = nullptr;

Shutdown_Capturing::Shutdown_Capturing(std::string const &ref_window_name_received) : _window_name(ref_window_name_received)
{
    if(this->Check_Systemd_Version() == false)
    {
        MyEA::String::Error("An error has been triggered from the `Check_Systemd_Version()` function.");

        this->Query_Shutdown();
    }
}

bool Shutdown_Capturing::Check_Systemd_Version(void) const
{
    std::string tmp_output_systemd_version(MyEA::String::Execute_Command("systemd --version"));

    if(tmp_output_systemd_version.empty())
    {
        MyEA::String::Error("An error has been triggered from the `Execute_Command(busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager PreparingForSleep)` function.");

        return(false);
    }

    std::string::size_type const tmp_character_position(tmp_output_systemd_version.find_first_of(" "));

    if(tmp_character_position == std::string::npos)
    {
        MyEA::String::Error("An error has been triggered from the `find_first_of( )` function.");

        return(false);
    }
        
    // Substring "systemd ".
    tmp_output_systemd_version = tmp_output_systemd_version.substr(tmp_character_position + 1);

    int tmp_systemd_version(0);
        
    try { tmp_systemd_version = std::stoi(tmp_output_systemd_version); }
    catch(std::exception &e)
    {
        MyEA::String::Error("An error has been triggered from the `std::stoi(%s) -> %s` function.", tmp_output_systemd_version.c_str(), e.what());

        return(false);
    }

    if(tmp_systemd_version < 220)
    {
        MyEA::String::Error("Systemd current version %d need to be update to the version 220 or greater.", tmp_systemd_version);

        return(false);
    }

    return(true);
}

void Shutdown_Capturing::Query_Shutdown(void)
{
    this->_on_shutdown = true;
        
    for(unsigned int tmp_boolean_index(0u); tmp_boolean_index != this->_number_boolean; ++tmp_boolean_index)
    {
        this->_ptr_array_ptr_shutdown_boolean[tmp_boolean_index]->store(true);
    }
}

void Shutdown_Capturing::Deallocate__Array_Shutdown_Boolean(void)
{
    SAFE_DELETE_ARRAY(this->_ptr_array_ptr_shutdown_boolean);
}

void Shutdown_Capturing::Initialize_Static_Shutdown_Block(void)
{
    ptr_global_Shutdown_Block = this;
}
    
bool Shutdown_Capturing::Get__On_Shutdown(void) const
{
    return(this->_on_shutdown);
}

bool Shutdown_Capturing::Create_Shutdown_Block(bool const use_ctrl_handler_received)
{
    if(this->_initialize == false)
    {
        this->Initialize_Static_Shutdown_Block();

        int tmp_return_code;

        // Connect to the system bus.
        if((tmp_return_code = sd_bus_open_system(&this->_ptr_sd_bus)) < 0)
        {
            MyEA::String::Error("An error has been triggered from the `sd_bus_open_system() -> %d` function.", tmp_return_code);

            sd_bus_error_free(&this->_sd_bus_error);

            return(false);
        }

        // Issue the method call and store the handle.
        if((tmp_return_code = sd_bus_call_method(this->_ptr_sd_bus,
                                                 "org.freedesktop.login1",         // Service to contact.
                                                 "/org/freedesktop/login1",        // Object path.
                                                 "org.freedesktop.login1.Manager", // Interface name.
                                                 "Inhibit",                        // Method name.
                                                 &this->_sd_bus_error,             // Object to return error in.
                                                 &this->_ptr_sd_bus_handle,        // Return message on success.
                                                 "ssss",                           // Input signature.
                                                 "shutdown",                       // argument: what.
                                                 this->_window_name.c_str(),       // argument: who.
                                                 "Closing in progress...",         // argument: why.
                                                 "delay")) < 0)                    // argument: mode.
        {
            MyEA::String::Error("An error has been triggered from the `sd_bus_call_method() -> (%d, %s)` function.",
                                tmp_return_code,
                                this->_sd_bus_error.message);

            sd_bus_unref(this->_ptr_sd_bus);
                
            sd_bus_error_free(&this->_sd_bus_error);

            return(false);
        }

        this->_initialize = true;

        return(true);
    }
    else { return(false); }
}

bool Shutdown_Capturing::Remove_Shutdown_Block(void)
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

bool Shutdown_Capturing::Peak_Message(void)
{
    if(this->_initialize && this->_on_shutdown == false)
    {
        int tmp_return_code,
            tmp_preparing_for_shutdown;

        // Issue the method call and store the value in a boolean.
        if((tmp_return_code = sd_bus_get_property(this->_ptr_sd_bus,
                                                  "org.freedesktop.login1",         // Service to contact.
                                                  "/org/freedesktop/login1",        // Object path.
                                                  "org.freedesktop.login1.Manager", // Interface name.
                                                  "PreparingForShutdown",           // Method name.
                                                  &this->_sd_bus_error,             // Object to return error in.
                                                  &this->_ptr_sd_bus_message,       // Reply.
                                                  "b"                               // Input signature.
                                                  )) < 0)                           // Output.
        {
            MyEA::String::Error("An error has been triggered from the `sd_bus_get_property() -> (%d, %s)` function.",
                                tmp_return_code,
                                this->_sd_bus_error.message);

            sd_bus_error_free(&this->_sd_bus_error);

            return(false);
        }
        
        // Read.
        if((tmp_return_code = sd_bus_message_read(this->_ptr_sd_bus_message, "b", &tmp_preparing_for_shutdown)) < 0)
        {
            MyEA::String::Error("An error has been triggered from the `sd_bus_message_read() -> %d` function.", tmp_return_code);

            return(false);
        }
        
        if(tmp_preparing_for_shutdown != 0) { this->Query_Shutdown(); }
    }

    return(true);
}
    
bool Shutdown_Capturing::Peak_Message_Async(void)
{
    if(this->_initialize == false)
    {
        MyEA::String::Error("Shutdown capturing not initialized.");

        return(false);
    }
    else if(this->_asynchronous_mode)
    {
        MyEA::String::Error("Asynchronous mode is already enabled.");

        return(false);
    }

    if(this->_on_shutdown == false)
    {
        this->_asynchronous_mode = true;

        this->_asynchronous_thread = std::thread(&Shutdown_Capturing::_Peak_Message_Async, this);
    }

    return(true);
}
    
bool Shutdown_Capturing::_Peak_Message_Async(void)
{
    if(this->_initialize)
    {
        if(this->Peak_Message() == false)
        {
            MyEA::String::Error("An error has been triggered from the `Peak_Message()` function.");

            return(false);
        }

        while(this->_on_shutdown == false && this->_asynchronous_mode)
        {
            std::this_thread::sleep_for(std::chrono::seconds(3));

            if(this->Peak_Message() == false)
            {
                MyEA::String::Error("An error has been triggered from the `Peak_Message()` function.");

                return(false);
            }
        }
    }

    return(true);
}
    
bool Shutdown_Capturing::Push_Back(std::atomic<bool> *const ptr_shutdown_boolean)
{
    if(this->_ptr_array_ptr_shutdown_boolean == nullptr)
    {
        std::atomic<bool> **tmp_ptr_array_ptr_shutdown_boolean;

        if((tmp_ptr_array_ptr_shutdown_boolean = new std::atomic<bool>*[1u]) == nullptr)
        {
            MyEA::String::Error("Cannot allocate %zu bytes.", sizeof(std::atomic<bool>*));

            return(false);
        }

        this->_ptr_array_ptr_shutdown_boolean = tmp_ptr_array_ptr_shutdown_boolean;

        this->_ptr_array_ptr_shutdown_boolean[0u] = ptr_shutdown_boolean;

        ++this->_number_boolean;
    }
    else
    {
        this->_ptr_array_ptr_shutdown_boolean = MyEA::Memory::reallocate_pointers_array_cpp<std::atomic<bool>*>(this->_ptr_array_ptr_shutdown_boolean,
                                                                                                                this->_number_boolean + 1_zu,
                                                                                                                this->_number_boolean,
                                                                                                                true);

        if(this->_ptr_array_ptr_shutdown_boolean == nullptr)
        {
            MyEA::String::Error("Cannot allocate %zu bytes.", (this->_number_boolean + 1_zu) * sizeof(std::atomic<bool>*));

            return(false);
        }

        this->_ptr_array_ptr_shutdown_boolean[this->_number_boolean] = ptr_shutdown_boolean;

        ++this->_number_boolean;
    }

    return(true);
}

Shutdown_Capturing::~Shutdown_Capturing(void)
{
    this->Remove_Shutdown_Block();
        
    if(this->_ptr_sd_bus != NULL) { sd_bus_unref(this->_ptr_sd_bus); }

    sd_bus_error_free(&this->_sd_bus_error);

    this->Deallocate__Array_Shutdown_Boolean();
}