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
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>

namespace MyEA::Capturing
{
    class Shutdown
    {
        public:
            Shutdown(std::string const &ref_window_name_received, std::string const &ref_class_name_received);

            ~Shutdown(void);

            void Initialize_Static_Shutdown_Block(void);

            void ConsoleCtrlHandler(DWORD const dwCtrlType_received);

            void Query_Shutdown(void);

            void Deallocate__Array_Shutdown_Boolean(void);

            bool Get__On_Shutdown(void) const;

            bool Create_Shutdown_Block(bool const use_ctrl_handler_received);

            bool Remove_Shutdown_Block(void);

            bool Peak_Message(void);

            bool Peak_Message_Async(void);

            bool Push_Back(std::atomic<bool> *const ptr_shutdown_boolean);

            ATOM Register_Class(HINSTANCE HINSTANCE_received);

            BOOL Init_Instance(HINSTANCE HINSTANCE_received, int nCmdShow_received);

            static BOOL WINAPI

            WINAPI__ConsoleCtrlHandler(DWORD

            dwCtrlType);

        private:
            std::atomic<bool> **_ptr_array_ptr_shutdown_boolean = nullptr;

            bool _asynchronous_mode = false;

            bool _initialize = false;

            bool _on_shutdown = false;

            bool _Peak_Message_Async(void);

            size_t _number_boolean = 0;

            std::string _window_name = "";

            std::string _class_name = "";

            HWND _HWND = NULL;

            HINSTANCE _HINSTANCE = NULL;

            MSG _MSG;

            std::thread _asynchronous_thread;
    };

    extern class Shutdown *ptr_global_Shutdown_Block;
}