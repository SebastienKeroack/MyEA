#include "stdafx.hpp"

// This.
#include <Capturing/Shutdown/Windows/Shutdown.hpp>
#include <Configuration/Configuration.hpp>
#include <Reallocate/Reallocate.hpp>
#include <Strings/String.hpp>

namespace MyEA::Capturing
{
    class Shutdown *ptr_global_Shutdown_Block = nullptr;

    Shutdown::Shutdown(std::string const &ref_window_name_received, std::string const &ref_class_name_received) : _window_name(ref_window_name_received),
                                                                                                                                      _class_name(ref_class_name_received)
    { }

    BOOL WINAPI Shutdown::WINAPI__ConsoleCtrlHandler(DWORD dwCtrlType_received)
    {
        if(dwCtrlType_received == CTRL_C_EVENT ||
           dwCtrlType_received == CTRL_BREAK_EVENT ||
           dwCtrlType_received == CTRL_CLOSE_EVENT)
        {
            if(ptr_global_Shutdown_Block != nullptr)
            {
                ptr_global_Shutdown_Block->ConsoleCtrlHandler(dwCtrlType_received);
            }

            return(TRUE);
        }

        return(FALSE);
    }

    void Shutdown::ConsoleCtrlHandler(DWORD const dwCtrlType_received)
    {
        if(this->_initialize)
        {
            if(dwCtrlType_received == CTRL_C_EVENT ||
               dwCtrlType_received == CTRL_BREAK_EVENT ||
               dwCtrlType_received == CTRL_CLOSE_EVENT)
            {
                SendMessage(this->_HWND,
                            WM_CLOSE,
                            0,
                            0);
            }
        }
    }

    LRESULT CALLBACK CallBack_WndProc(HWND HWND_received,
                                      UINT message_received,
                                      WPARAM wParam_received,
                                      LPARAM lParam_received)
    {
        PAINTSTRUCT tmp_PAINT_STRUCT;

        HDC tmp_HDC;

        switch(message_received)
        {
            //case WM_CLOSE: return(FALSE); // 16
            //case WM_QUIT: return(FALSE); // 18
            //case WM_ENDSESSION: return(FALSE); // 22
            //case WM_TIMECHANGE: return(FALSE); // 30
            //case SPI_SETDOUBLECLKHEIGHT: return(FALSE); // 30
            //case VK_ACCEPT: return(FALSE); // 30
            //case WM_GETMINMAXINFO: return(FALSE); // 36
            //case SPI_SETSTICKYKEYS: return(FALSE); // 59
            //case WM_GETICON: return(FALSE); // 127
            //case WM_NCCREATE: return(FALSE); // 129
            //case WM_NCDESTROY: return(FALSE); // 130
            //case SPI_GETWINARRANGING: return(FALSE); // 130
            //case VK_F19: return(FALSE); // 130
            //case CF_DSPBITMAP: return(FALSE); // 130
            //case WM_NCCALCSIZE: return(FALSE); // 131
            //case WM_NCACTIVATE: return(FALSE); // 134
            //case SPI_GETDOCKMOVING: return(FALSE); // 144
            //case VK_NUMLOCK: return(FALSE); // 144
            //case WM_SYSTIMER: return(FALSE); // 280
            //case WM_DEVICECHANGE: return(FALSE); // 537
            //case WM_DWMNCRENDERINGCHANGED: return(FALSE); // 799
            //case WM_DWMCOLORIZATIONCOLORCHANGED: return(FALSE); // 800



            case WM_CREATE: // 1
                ShutdownBlockReasonCreate(HWND_received, L"Closing in progress...");
                    break;
            case WM_DESTROY: // 2
                PostQuitMessage(0);
                    break;
            case WM_PAINT: // 15
                tmp_HDC = BeginPaint(HWND_received, &tmp_PAINT_STRUCT);
                EndPaint(HWND_received, &tmp_PAINT_STRUCT);
                    break;
            case WM_QUERYENDSESSION: // 17
                if(ptr_global_Shutdown_Block != nullptr)
                {
                    ptr_global_Shutdown_Block->Query_Shutdown();
                }
                    return(FALSE);
            default:
                return(DefWindowProc(HWND_received,
                                     message_received,
                                     wParam_received,
                                     lParam_received));
        }

        return(0);
    }

    ATOM Shutdown::Register_Class(HINSTANCE HINSTANCE_received)
    {
        WNDCLASSEX tmp_WNDCLASSEX;

        tmp_WNDCLASSEX.cbSize = sizeof(WNDCLASSEX);
        tmp_WNDCLASSEX.style = CS_HREDRAW | CS_VREDRAW;
        tmp_WNDCLASSEX.lpfnWndProc = CallBack_WndProc;
        tmp_WNDCLASSEX.cbClsExtra = 0;
        tmp_WNDCLASSEX.cbWndExtra = 0;
        tmp_WNDCLASSEX.hInstance = HINSTANCE_received;
        tmp_WNDCLASSEX.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        tmp_WNDCLASSEX.hCursor = LoadCursor(NULL, IDC_ARROW);
        tmp_WNDCLASSEX.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        tmp_WNDCLASSEX.lpszMenuName = NULL;
        tmp_WNDCLASSEX.lpszClassName = this->_class_name.c_str();
        tmp_WNDCLASSEX.hIconSm = NULL;

        return(RegisterClassEx(&tmp_WNDCLASSEX));
    }

    BOOL Shutdown::Init_Instance(HINSTANCE HINSTANCE_received, int nCmdShow_received)
    {
        if(this->_window_name.empty())
        {
            MyEA::String::Error("Window name is empty.");

            return(FALSE);
        }
        else if(this->_class_name.empty())
        {
            MyEA::String::Error("Class name is empty.");

            return(FALSE);
        }

        BOOL tmp_return_code(0);

        RECT tmp_RECT = {0, 0, 512, 512};

        AdjustWindowRect(&tmp_RECT,
                         WS_OVERLAPPEDWINDOW,
                         TRUE);

        this->_HWND = CreateWindow(this->_class_name.c_str(),
                                   this->_window_name.c_str(),
                                   WS_OVERLAPPEDWINDOW,
                                   CW_USEDEFAULT,
                                   CW_USEDEFAULT,
                                   tmp_RECT.right - tmp_RECT.left,
                                   tmp_RECT.bottom - tmp_RECT.top,
                                   NULL,
                                   NULL,
                                   HINSTANCE_received,
                                   NULL);

        if(this->_HWND == NULL)
        {
            MyEA::String::Error("An error has been triggered from the `CreateWindow()` function.");

            return(FALSE);
        }

        ShowWindow(this->_HWND, nCmdShow_received);

        if((tmp_return_code = UpdateWindow(this->_HWND)) == FALSE)
        {
            MyEA::String::Error("An error has been triggered from the `UpdateWindow() -> %d` function.", tmp_return_code);

            return(FALSE);
        }

        return(TRUE);
    }

    void Shutdown::Query_Shutdown(void)
    {
        this->_on_shutdown = true;

        for(unsigned int tmp_boolean_index(0u); tmp_boolean_index != this->_number_boolean; ++tmp_boolean_index)
        {
            this->_ptr_array_ptr_shutdown_boolean[tmp_boolean_index]->store(true);
        }
    }

    void Shutdown::Deallocate__Array_Shutdown_Boolean(void)
    {
        SAFE_DELETE_ARRAY(this->_ptr_array_ptr_shutdown_boolean);
    }

    void Shutdown::Initialize_Static_Shutdown_Block(void)
    {
        ptr_global_Shutdown_Block = this;
    }

    bool Shutdown::Get__On_Shutdown(void) const
    {
        return(this->_on_shutdown);
    }

    bool Shutdown::Create_Shutdown_Block(bool const use_ctrl_handler_received)
    {
        if(this->_initialize == false)
        {
            BOOL tmp_return_code(0);

            this->Initialize_Static_Shutdown_Block();

            if(this->_HINSTANCE == NULL)
            {
                if(use_ctrl_handler_received && (tmp_return_code = SetConsoleCtrlHandler(static_cast<PHANDLER_ROUTINE>(Shutdown::WINAPI__ConsoleCtrlHandler), TRUE)) == FALSE)
                {
                    MyEA::String::Error("An error has been triggered from the `SetConsoleCtrlHandler() -> %d` function.", tmp_return_code);

                    return(false);
                }

                HINSTANCE tmp_HINSTANCE(GetModuleHandle(NULL));

                if(tmp_HINSTANCE == NULL)
                {
                    MyEA::String::Error("An error has been triggered from the `GetModuleHandle()` function.");

                    return(false);
                }

                this->Register_Class(tmp_HINSTANCE);

                if((tmp_return_code = this->Init_Instance(tmp_HINSTANCE, SW_HIDE)) == FALSE)
                {
                    MyEA::String::Error("An error has been triggered from the `Init_Instance() -> %d` function.", tmp_return_code);

                    return(false);
                }

                this->_HINSTANCE = tmp_HINSTANCE;
            }
            else
            {
                if(this->_HWND == NULL)
                {
                    MyEA::String::Error("HWD is a nullptr while HINSTANCE is not a nullptr.");

                    return(false);
                }

                ShutdownBlockReasonCreate(this->_HWND, L"Closing in progress...");
            }

            this->_initialize = true;

            return(true);
        }
        else { return(false); }
    }

    bool Shutdown::Remove_Shutdown_Block(void)
    {
        if(this->_initialize)
        {
            if(this->_asynchronous_mode)
            {
                this->_asynchronous_mode = false;

                if(this->_asynchronous_thread.joinable()) { this->_asynchronous_thread.join(); }
            }

            ShutdownBlockReasonDestroy(this->_HWND);

            this->_initialize = false;

            return(true);
        }
        else { return(false); }
    }

    bool Shutdown::Peak_Message(void)
    {
        if(this->_initialize && this->_on_shutdown == false)
        {
            BOOL tmp_return_code(0);

            while((tmp_return_code = PeekMessage(&this->_MSG,
                                                 this->_HWND,
                                                 0,
                                                 0,
                                                 PM_REMOVE)) != 0)
            {
                if(tmp_return_code == -1)
                {
                    MyEA::String::Error("An error has been triggered from the `PeekMessage() -> %d` function.", tmp_return_code);

                    return(false);
                }

                TranslateMessage(&this->_MSG);

                DispatchMessage(&this->_MSG);
            }

            if(this->_MSG.message == WM_QUIT) { this->Query_Shutdown(); }
        }

        return(true);
    }

    bool Shutdown::Peak_Message_Async(void)
    {
        if(this->_initialize == false)
        {
            MyEA::String::Error("Shutdown block not initialized.");

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

            this->_asynchronous_thread = std::thread(&Shutdown::_Peak_Message_Async, this);
        }

        return(true);
    }

    bool Shutdown::_Peak_Message_Async(void)
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

    bool Shutdown::Push_Back(std::atomic<bool> *const ptr_shutdown_boolean)
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

    Shutdown::~Shutdown(void)
    {
        this->Remove_Shutdown_Block();

        this->Deallocate__Array_Shutdown_Boolean();
}
}