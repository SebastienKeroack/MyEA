#include "stdafx.hpp"

#include <Form.hpp>
#include <Form/Form__Neural_Network.h>

#include <Files/File.hpp>

#include <msclr\auto_gcroot.h>

namespace MyEA
{
    namespace Form
    {
        template<typename FORM>
        struct struct_Thread_Form
        {
            bool Allocate(void);
            bool Deallocate(void);
            
            static int Run(FORM^ ptr_FORM_received);

            HANDLE handle_thread = NULL;
            HANDLE handle_initialized = NULL;
            
            msclr::auto_gcroot<FORM^> ptr_GC_Form = nullptr;
        };

        template<typename FORM>
        DWORD THREAD_FORM_START_ROUTINE(LPVOID lp_struct_Thread_Form)
        {
            struct struct_Thread_Form<FORM> *tmp_ptr_struct_Thread_Form(reinterpret_cast<struct struct_Thread_Form<FORM>*>(lp_struct_Thread_Form));
            
            tmp_ptr_struct_Thread_Form->ptr_GC_Form = gcnew FORM();
            
            // Telling the thread is initialized.
            SetEvent(tmp_ptr_struct_Thread_Form->handle_initialized);

            struct_Thread_Form<FORM>::Run(tmp_ptr_struct_Thread_Form->ptr_GC_Form.get());

            tmp_ptr_struct_Thread_Form->ptr_GC_Form.reset();

            return(0ul);
        }

        template<typename FORM>
        int struct_Thread_Form<FORM>::Run(FORM^ ptr_FORM_received)
        {
            // Enabling Windows XP visual effects before any controls are created
            //Application::EnableVisualStyles();
            //Application::SetCompatibleTextRenderingDefault(false);

            Application::Run(ptr_FORM_received);

            return(0);
        }
        
        template<typename FORM>
        bool struct_Thread_Form<FORM>::Allocate(void)
        {
            if(safe_cast<FORM^>(this->ptr_GC_Form.get()) == nullptr)
            {
                // Create event for telling if the thread is initialized.
                this->handle_initialized = CreateEvent(NULL, FALSE, FALSE, NULL);

                // Create a thread containing the form.
                if((this->handle_thread = CreateThread(NULL,
                                                                        0,
                                                                        reinterpret_cast<LPTHREAD_START_ROUTINE>(THREAD_FORM_START_ROUTINE<FORM>),
                                                                        reinterpret_cast<LPVOID>(this),
                                                                        0,
                                                                        NULL)) == NULL)
                { return(false); }

                // Waiting thread to initialize.
                WaitForSingleObject(this->handle_initialized, INFINITE);

                // Initialization finish. Close the handle (event).
                CloseHandle(this->handle_initialized);
            }
            else { return(false); }

            return(true);
        }
        
        template<typename FORM>
        bool struct_Thread_Form<FORM>::Deallocate(void)
        {
            if(safe_cast<FORM^>(this->ptr_GC_Form.get()) != nullptr)
            {
                // Close form.
                this->ptr_GC_Form->Close();

                // Waiting thread
                WaitForSingleObject(this->handle_thread, INFINITE);

                // Close thread
                CloseHandle(this->handle_thread);

                return(true);
            }

            return(false);
        }

        struct struct_Thread_Form<Form_Neural_Network> *global_ptr_Form_Neural_Network = nullptr;

        DLL_API bool DLL_API API__Form__Is_Loaded(void) { return(true); }

        DLL_API void DLL_API API__Form__Neural_Network__Allocate(void)
        {
            if(global_ptr_Form_Neural_Network == nullptr)
            {
                global_ptr_Form_Neural_Network = new struct struct_Thread_Form<Form_Neural_Network>;

                if(global_ptr_Form_Neural_Network->Allocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: Can not allocate 'global_ptr_Form_Neural_Network'." NEW_LINE, __FUNCTION__);

                    SAFE_DELETE(global_ptr_Form_Neural_Network);
                }
            }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Use_Datapoint_Training(bool const use_datapoint_training_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            { global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Use_Datapoint_Training(use_datapoint_training_received); }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Initialize(unsigned int const type_chart_received, unsigned int const number_series_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            { global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Initialize(type_chart_received, number_series_received); }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Total_Means(unsigned int const total_means_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            { global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Total_Means(total_means_received); }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Reset(unsigned int const type_chart_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            { global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Reset(type_chart_received); }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Rescale(unsigned int const type_chart_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            { global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Rescale(type_chart_received); }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Add_Point(unsigned int const type_chart_received,
                                                                                                                            unsigned int const index_series_received,
                                                                                                                            unsigned int const type_loss_received,
                                                                                                                            double const x_received,
                                                                                                                            double const y_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Add_Point(type_chart_received,
                                                                                                                  index_series_received,
                                                                                                                  type_loss_received,
                                                                                                                  x_received,
                                                                                                                  y_received);
            }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Grid_Search_Add_Column(std::string const &ref_value_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Grid_Search_Add_Column(gcnew System::String(ref_value_received.c_str()));
            }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Grid_Search_Add_Row(unsigned int const cell_index_received, std::string const &ref_value_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Grid_Search_Add_Row(cell_index_received, gcnew System::String(ref_value_received.c_str()));
            }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Loss_Diff(unsigned int const index_series_received,
                                                                                                                           unsigned int const type_received,
                                                                                                                           double const x_received)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Loss_Diff(index_series_received,
                                                                                                                 type_received,
                                                                                                                 x_received);
            }
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Chart_Scale(unsigned int const type_chart_received,
                                                                                                                      unsigned int const index_series_received,
                                                                                                                      unsigned int const type_loss_received,
                                                                                                                      bool const scale_y_axis_received,
                                                                                                                      double const x_received,
                                                                                                                      double const y_received)
        {
            /*
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->ptr_GC_Form->Chart_Add_Point(type_chart_received,
                                                                                                                  index_series_received,
                                                                                                                  type_loss_received,
                                                                                                                  x_received,
                                                                                                                  y_received);
            }
            */
        }
        
        DLL_API void DLL_API API__Form__Neural_Network__Deallocate(void)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->Deallocate();

                SAFE_DELETE(global_ptr_Form_Neural_Network);
            }
        }
        
        DLL_API bool DLL_API API__Form__Neural_Network__Get_Signal_Training_Stop(void)
        {
            if(global_ptr_Form_Neural_Network != nullptr) { return(global_ptr_Form_Neural_Network->ptr_GC_Form->Get__Signal_Training_Stop()); }

            return(false);
        }
        
        DLL_API bool DLL_API API__Form__Neural_Network__Get_Signal_Training_Menu(void)
        {
            if(global_ptr_Form_Neural_Network != nullptr) { return(global_ptr_Form_Neural_Network->ptr_GC_Form->Get__Signal_Training_Menu()); }

            return(false);
        }
        
        DLL_API bool DLL_API API__Form__Neural_Network__Reset_Signal_Training_Menu(void)
        {
            if(global_ptr_Form_Neural_Network != nullptr)
            {
                global_ptr_Form_Neural_Network->ptr_GC_Form->Reset__Signal_Training_Menu();

                return(true);
            }

            return(false);
        }
    }
}