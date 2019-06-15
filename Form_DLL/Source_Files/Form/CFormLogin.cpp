#include "stdafx.hpp"

#include <Form/CFormLogin.h>

namespace MyEA
{
    namespace Form
    {
        /* MAIN_CFNN */
        const int MAIN_CFL(CFormLogin^ ptr_CFL_received)
        {
            // Enabling Windows XP visual effects before any controls are created
            Application::EnableVisualStyles();
            Application::SetCompatibleTextRenderingDefault(false);

            Application::Run(ptr_CFL_received);

            return(0);
        }

        CFormLogin::CFormLogin(void) : isThreading(true),
                                                            _try_connect(false),
                                                            _connect(false),
                                                            p_Ptr_Ressource_Manager(gcnew ResourceManager("FormWin.Resource_Files.Resource", GetType()->Assembly))
        {
            InitializeComponent();
            //
            //TODO: Add the constructor code here
            //

            this->Icon = safe_cast<System::Drawing::Icon^>(p_Ptr_Ressource_Manager->GetObject("COMPANY_favicon_64_ICO"));
            this->LOGIN_BUTTON_CLOSE->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Windows_App_Close_64_PNG"));
        }

        void CFormLogin::FormNN_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->WindowState != FormWindowState::Maximized)
            {
                this->_moving = true;
                this->_position_offset = Point(e->X, e->Y);
            }
        }
        void CFormLogin::FormMain_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->_moving)
            {
                Point currentScreenPos = PointToScreen(e->Location);
                this->Location = Point(currentScreenPos.X - this->_position_offset.X, currentScreenPos.Y - this->_position_offset.Y);
            }
        }
        void CFormLogin::FormMain_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_moving = false;
        }

        void CFormLogin::FormClose(void)
        {
            this->Close();
        }

        void CFormLogin::LOGIN_BUTTON_CLOSE_Click(Object^  sender, EventArgs^  e)
        {
            this->FormClose();
        }

        void CFormLogin::LOGIN_BUTTON_SIGN_IN_Click(Object^  sender, EventArgs^  e)
        {
            this->_try_connect = true;
            this->SignIn();
            //this->FormClose();
        }

        void CFormLogin::LOGIN_BUTTON_SIGN_IN_KeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e)
        {
            if(e->KeyCode == Keys::Return)
            {
                this->_try_connect = true;
                this->SignIn();
            }
        }

        const bool CFormLogin::SignIn(void)
        {
            bool tmp_boolean(false);
            
            if(this->_try_connect)
            {
                this->_try_connect = false;
                if(this->LoginUsernameTextBox->Text == "Username" &&
                   this->LoginPasswordTextBox->Text == "mypassword") { tmp_boolean = true; }
                else { this->LoginPasswordTextBox->Text = ""; }
            }

            return(_connect = tmp_boolean);
        }

        void CFormLogin::Exit(const bool exiTprocess_received)
        {
            this->isThreading = false;

            if(exiTprocess_received) { ExitThread(0); }
            else { Close(); }
        }

        CFormLogin::~CFormLogin()
        {
            if(p_Ptr_Ressource_Manager) { delete(p_Ptr_Ressource_Manager); }
            if(components) { delete(components); }
        }
    }
}