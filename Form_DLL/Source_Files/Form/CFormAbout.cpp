#include "stdafx.hpp"

#include <Form/CFormAbout.h>

namespace MyEA
{
    namespace Form
    {
        CFormAbout::CFormAbout(ResourceManager^ ptr_rm_received)
        {
            InitializeComponent();
            //
            //TODO: Add the constructor code here
            //
            _Ptr_Ressource_Manager = ptr_rm_received;

            this->Icon = safe_cast<System::Drawing::Icon^>(ptr_rm_received->GetObject("COMPANY_favicon_64_ICO"));
            this->ABOUTLOGO_PICTURE->BackgroundImage = safe_cast<Image^>(ptr_rm_received->GetObject("COMPANY_favicon_64_PNG"));
            this->ABOUTBUTTON_CLOSE->BackgroundImage = safe_cast<Image^>(ptr_rm_received->GetObject("Windows_App_Close_64_PNG"));
        }

        void CFormAbout::FormNN_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->WindowState != FormWindowState::Maximized)
            {
                this->_moving = true;
                this->_position_offset = Point(e->X, e->Y);
            }
        }
        void CFormAbout::FormMain_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->_moving)
            {
                Point currentScreenPos = PointToScreen(e->Location);
                this->Location = Point(currentScreenPos.X - this->_position_offset.X, currentScreenPos.Y - this->_position_offset.Y);
            }
        }
        void CFormAbout::FormMain_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_moving = false;
        }

        void CFormAbout::FormClose(void)
        {
            this->Close();
        }

        void CFormAbout::ABOUTBUTTON_CLOSE_Click(Object^  sender, EventArgs^  e)
        {
            this->FormClose();
        }

        void CFormAbout::ABOUTBUTTON_OK_Click(Object^  sender, EventArgs^  e)
        {
            this->FormClose();
        }

        CFormAbout::~CFormAbout()
        {
            if(_Ptr_Ressource_Manager) { delete(_Ptr_Ressource_Manager); }
            if(components) { delete(components); }
        }
    }
}