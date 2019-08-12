#pragma once

// Standard.
#include <string>
#include <thread>
#include <atomic>

// Systemd
#include <systemd/sd-bus.h> // sudo apt install libsystemd-dev

namespace MyEA::Capturing
{
    class Shutdown
    {
        public:
            Shutdown(std::string const &ref_window_name_received);

            ~Shutdown(void);

            void Initialize_Static_Shutdown_Block(void);

            void Query_Shutdown(void);

            void Deallocate__Array_Shutdown_Boolean(void);

            bool Check_Systemd_Version(void) const;

            bool Get__On_Shutdown(void) const;

            bool Create_Shutdown_Block(bool const use_ctrl_handler_received);

            bool Remove_Shutdown_Block(void);

            bool Peak_Message(void);

            bool Peak_Message_Async(void);

            bool Push_Back(std::atomic<bool> *const ptr_shutdown_boolean);

        private:
            std::atomic<bool> **_ptr_array_ptr_shutdown_boolean = nullptr;

            bool _asynchronous_mode = false;

            bool _initialize = false;

            bool _on_shutdown = false;

            bool _Peak_Message_Async(void);

            size_t _number_boolean = 0u;

            std::string _window_name = "";

            std::string _class_name = "";

            sd_bus *_ptr_sd_bus = NULL;

            sd_bus_message *_ptr_sd_bus_handle = NULL;

            sd_bus_message *_ptr_sd_bus_message = NULL;

            sd_bus_error _sd_bus_error = SD_BUS_ERROR_NULL;

            std::thread _asynchronous_thread;
    };

    extern class Shutdown *ptr_global_Shutdown_Block;
}