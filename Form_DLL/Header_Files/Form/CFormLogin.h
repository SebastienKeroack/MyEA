#pragma once

#include <msclr\marshal_cppstd.h>

namespace MyEA
{
    namespace Form
    {
        using namespace System;
        using namespace System::Resources;
        using namespace System::ComponentModel;
        using namespace System::Collections;
        using namespace System::Windows::Forms;
        using namespace System::Data;
        using namespace System::Drawing;

        /// <summary>
        /// Summary for CFormLogin
        /// </summary>
        public ref class CFormLogin : public System::Windows::Forms::Form
        {
        protected:
            System::Resources::ResourceManager^ p_Ptr_Ressource_Manager;

        public:
            CFormLogin(System::Void);

            System::Void FormClose(System::Void);
            System::Void Exit(const System::Boolean exiTprocess_received);
            
            const System::Boolean SignIn(System::Void);
            const System::Boolean Get__Connected(System::Void) { return(_connect); }

            System::Boolean isThreading;

        private:
            System::Boolean _moving;
            System::Boolean _try_connect;
            System::Boolean _connect;

            private: System::Windows::Forms::TableLayoutPanel^  LoginFormLayoutPanel;
            private: System::Windows::Forms::TextBox^  LoginPasswordTextBox;
            private: System::Windows::Forms::TextBox^  LoginUsernameTextBox;
            private: System::Windows::Forms::Label^  LOGIN_LABEL;

            System::Drawing::Point _position_offset;

        protected:
            /// <summary>
            /// Clean up any resources being used.
            /// </summary>
            ~CFormLogin();
            private: System::Windows::Forms::TableLayoutPanel^  LoginSecondaryTableLayoutPanel;
            protected:


            private: System::Windows::Forms::TableLayoutPanel^  LoginTopTableLayoutPanel;



            private: System::Windows::Forms::Button^  LOGIN_BUTTON_CLOSE;
            private: System::Windows::Forms::Button^  LOGIN_BUTTON_SIGN_IN;






            private: System::Windows::Forms::Label^  LOGIN_TITLE_LABEL;
            private: System::Windows::Forms::TableLayoutPanel^  LoginPrimaryTableLayoutPanel;







        private:
            /// <summary>
            /// Required designer variable.
            /// </summary>
            System::ComponentModel::Container ^components;

    #pragma region Windows Form Designer generated code
            /// <summary>
            /// Required method for Designer support - do not modify
            /// the contents of this method with the code editor.
            /// </summary>
            void InitializeComponent(void)
            {
                this->LoginSecondaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->LoginTopTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->LOGIN_BUTTON_CLOSE = (gcnew System::Windows::Forms::Button());
                this->LOGIN_TITLE_LABEL = (gcnew System::Windows::Forms::Label());
                this->LOGIN_BUTTON_SIGN_IN = (gcnew System::Windows::Forms::Button());
                this->LoginFormLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->LoginPasswordTextBox = (gcnew System::Windows::Forms::TextBox());
                this->LoginUsernameTextBox = (gcnew System::Windows::Forms::TextBox());
                this->LOGIN_LABEL = (gcnew System::Windows::Forms::Label());
                this->LoginPrimaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->LoginSecondaryTableLayoutPanel->SuspendLayout();
                this->LoginTopTableLayoutPanel->SuspendLayout();
                this->LoginFormLayoutPanel->SuspendLayout();
                this->LoginPrimaryTableLayoutPanel->SuspendLayout();
                this->SuspendLayout();
                // 
                // LoginSecondaryTableLayoutPanel
                // 
                this->LoginSecondaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LoginSecondaryTableLayoutPanel->ColumnCount = 1;
                this->LoginSecondaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginSecondaryTableLayoutPanel->Controls->Add(this->LoginTopTableLayoutPanel, 0, 0);
                this->LoginSecondaryTableLayoutPanel->Controls->Add(this->LOGIN_BUTTON_SIGN_IN, 0, 2);
                this->LoginSecondaryTableLayoutPanel->Controls->Add(this->LoginFormLayoutPanel, 0, 1);
                this->LoginSecondaryTableLayoutPanel->Location = System::Drawing::Point(1, 1);
                this->LoginSecondaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->LoginSecondaryTableLayoutPanel->Name = L"LoginSecondaryTableLayoutPanel";
                this->LoginSecondaryTableLayoutPanel->RowCount = 3;
                this->LoginSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->LoginSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    50)));
                this->LoginSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->LoginSecondaryTableLayoutPanel->Size = System::Drawing::Size(598, 198);
                this->LoginSecondaryTableLayoutPanel->TabIndex = 1;
                // 
                // LoginTopTableLayoutPanel
                // 
                this->LoginTopTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LoginTopTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->LoginTopTableLayoutPanel->ColumnCount = 3;
                this->LoginTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    125)));
                this->LoginTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->LoginTopTableLayoutPanel->Controls->Add(this->LOGIN_BUTTON_CLOSE, 2, 0);
                this->LoginTopTableLayoutPanel->Controls->Add(this->LOGIN_TITLE_LABEL, 0, 0);
                this->LoginTopTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->LoginTopTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->LoginTopTableLayoutPanel->Name = L"LoginTopTableLayoutPanel";
                this->LoginTopTableLayoutPanel->RowCount = 1;
                this->LoginTopTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginTopTableLayoutPanel->Size = System::Drawing::Size(598, 30);
                this->LoginTopTableLayoutPanel->TabIndex = 1;
                this->LoginTopTableLayoutPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormLogin::FormNN_Move_MouseDown);
                this->LoginTopTableLayoutPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormLogin::FormMain_Move_MouseMove);
                this->LoginTopTableLayoutPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormLogin::FormMain_Move_MouseUp);
                // 
                // LOGIN_BUTTON_CLOSE
                // 
                this->LOGIN_BUTTON_CLOSE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LOGIN_BUTTON_CLOSE->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->LOGIN_BUTTON_CLOSE->FlatAppearance->BorderSize = 0;
                this->LOGIN_BUTTON_CLOSE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->LOGIN_BUTTON_CLOSE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->LOGIN_BUTTON_CLOSE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->LOGIN_BUTTON_CLOSE->Location = System::Drawing::Point(568, 0);
                this->LOGIN_BUTTON_CLOSE->Margin = System::Windows::Forms::Padding(0);
                this->LOGIN_BUTTON_CLOSE->Name = L"LOGIN_BUTTON_CLOSE";
                this->LOGIN_BUTTON_CLOSE->Size = System::Drawing::Size(30, 30);
                this->LOGIN_BUTTON_CLOSE->TabIndex = 3;
                this->LOGIN_BUTTON_CLOSE->UseVisualStyleBackColor = false;
                this->LOGIN_BUTTON_CLOSE->Click += gcnew System::EventHandler(this, &CFormLogin::LOGIN_BUTTON_CLOSE_Click);
                // 
                // LOGIN_TITLE_LABEL
                // 
                this->LOGIN_TITLE_LABEL->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LOGIN_TITLE_LABEL->Font = (gcnew System::Drawing::Font(L"Arial", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->LOGIN_TITLE_LABEL->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(153)),
                                                                                      static_cast<System::Int32>(static_cast<System::Byte>(153)), static_cast<System::Int32>(static_cast<System::Byte>(153)));
                this->LOGIN_TITLE_LABEL->Location = System::Drawing::Point(0, 0);
                this->LOGIN_TITLE_LABEL->Margin = System::Windows::Forms::Padding(0);
                this->LOGIN_TITLE_LABEL->Name = L"LOGIN_TITLE_LABEL";
                this->LOGIN_TITLE_LABEL->Size = System::Drawing::Size(125, 30);
                this->LOGIN_TITLE_LABEL->TabIndex = 1;
                this->LOGIN_TITLE_LABEL->Text = L"MyEA - Login";
                this->LOGIN_TITLE_LABEL->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                this->LOGIN_TITLE_LABEL->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormLogin::FormNN_Move_MouseDown);
                this->LOGIN_TITLE_LABEL->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormLogin::FormMain_Move_MouseMove);
                this->LOGIN_TITLE_LABEL->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormLogin::FormMain_Move_MouseUp);
                // 
                // LOGIN_BUTTON_SIGN_IN
                // 
                this->LOGIN_BUTTON_SIGN_IN->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom));
                this->LOGIN_BUTTON_SIGN_IN->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->LOGIN_BUTTON_SIGN_IN->FlatAppearance->BorderColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                           static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->LOGIN_BUTTON_SIGN_IN->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                  static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->LOGIN_BUTTON_SIGN_IN->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                                  static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->LOGIN_BUTTON_SIGN_IN->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->LOGIN_BUTTON_SIGN_IN->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->LOGIN_BUTTON_SIGN_IN->ForeColor = System::Drawing::Color::White;
                this->LOGIN_BUTTON_SIGN_IN->Location = System::Drawing::Point(199, 158);
                this->LOGIN_BUTTON_SIGN_IN->Margin = System::Windows::Forms::Padding(10);
                this->LOGIN_BUTTON_SIGN_IN->Name = L"LOGIN_BUTTON_SIGN_IN";
                this->LOGIN_BUTTON_SIGN_IN->Size = System::Drawing::Size(200, 30);
                this->LOGIN_BUTTON_SIGN_IN->TabIndex = 2;
                this->LOGIN_BUTTON_SIGN_IN->Text = L"Sign In";
                this->LOGIN_BUTTON_SIGN_IN->UseVisualStyleBackColor = false;
                this->LOGIN_BUTTON_SIGN_IN->Click += gcnew System::EventHandler(this, &CFormLogin::LOGIN_BUTTON_SIGN_IN_Click);
                this->LOGIN_BUTTON_SIGN_IN->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &CFormLogin::LOGIN_BUTTON_SIGN_IN_KeyUp);
                // 
                // LoginFormLayoutPanel
                // 
                this->LoginFormLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LoginFormLayoutPanel->ColumnCount = 1;
                this->LoginFormLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginFormLayoutPanel->Controls->Add(this->LoginPasswordTextBox, 0, 2);
                this->LoginFormLayoutPanel->Controls->Add(this->LoginUsernameTextBox, 0, 1);
                this->LoginFormLayoutPanel->Controls->Add(this->LOGIN_LABEL, 0, 0);
                this->LoginFormLayoutPanel->Location = System::Drawing::Point(0, 30);
                this->LoginFormLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->LoginFormLayoutPanel->Name = L"LoginFormLayoutPanel";
                this->LoginFormLayoutPanel->RowCount = 3;
                this->LoginFormLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginFormLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    40)));
                this->LoginFormLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    40)));
                this->LoginFormLayoutPanel->Size = System::Drawing::Size(598, 118);
                this->LoginFormLayoutPanel->TabIndex = 5;
                // 
                // LoginPasswordTextBox
                // 
                this->LoginPasswordTextBox->Anchor = System::Windows::Forms::AnchorStyles::Top;
                this->LoginPasswordTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->LoginPasswordTextBox->Font = (gcnew System::Drawing::Font(L"Arial", 11.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->LoginPasswordTextBox->Location = System::Drawing::Point(129, 89);
                this->LoginPasswordTextBox->Margin = System::Windows::Forms::Padding(0, 11, 0, 0);
                this->LoginPasswordTextBox->MaxLength = 255;
                this->LoginPasswordTextBox->Name = L"LoginPasswordTextBox";
                this->LoginPasswordTextBox->PasswordChar = '*';
                this->LoginPasswordTextBox->Size = System::Drawing::Size(340, 18);
                this->LoginPasswordTextBox->TabIndex = 1;
                this->LoginPasswordTextBox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
                this->LoginPasswordTextBox->UseSystemPasswordChar = true;
                this->LoginPasswordTextBox->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &CFormLogin::LOGIN_BUTTON_SIGN_IN_KeyUp);
                // 
                // LoginUsernameTextBox
                // 
                this->LoginUsernameTextBox->Anchor = System::Windows::Forms::AnchorStyles::Top;
                this->LoginUsernameTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->LoginUsernameTextBox->Font = (gcnew System::Drawing::Font(L"Arial", 11.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->LoginUsernameTextBox->Location = System::Drawing::Point(129, 49);
                this->LoginUsernameTextBox->Margin = System::Windows::Forms::Padding(0, 11, 0, 0);
                this->LoginUsernameTextBox->MaxLength = 255;
                this->LoginUsernameTextBox->Name = L"LoginUsernameTextBox";
                this->LoginUsernameTextBox->Size = System::Drawing::Size(340, 18);
                this->LoginUsernameTextBox->TabIndex = 0;
                this->LoginUsernameTextBox->Text = L"Username";
                this->LoginUsernameTextBox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
                this->LoginUsernameTextBox->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &CFormLogin::LOGIN_BUTTON_SIGN_IN_KeyUp);
                // 
                // LOGIN_LABEL
                // 
                this->LOGIN_LABEL->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LOGIN_LABEL->Font = (gcnew System::Drawing::Font(L"Arial", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->LOGIN_LABEL->ForeColor = System::Drawing::Color::White;
                this->LOGIN_LABEL->Location = System::Drawing::Point(0, 0);
                this->LOGIN_LABEL->Margin = System::Windows::Forms::Padding(0);
                this->LOGIN_LABEL->Name = L"LOGIN_LABEL";
                this->LOGIN_LABEL->Size = System::Drawing::Size(598, 38);
                this->LOGIN_LABEL->TabIndex = 2;
                this->LOGIN_LABEL->Text = L"Login";
                this->LOGIN_LABEL->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
                // 
                // LoginPrimaryTableLayoutPanel
                // 
                this->LoginPrimaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LoginPrimaryTableLayoutPanel->ColumnCount = 3;
                this->LoginPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->LoginPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->LoginPrimaryTableLayoutPanel->Controls->Add(this->LoginSecondaryTableLayoutPanel, 1, 1);
                this->LoginPrimaryTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->LoginPrimaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->LoginPrimaryTableLayoutPanel->Name = L"LoginPrimaryTableLayoutPanel";
                this->LoginPrimaryTableLayoutPanel->RowCount = 3;
                this->LoginPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->LoginPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->LoginPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->LoginPrimaryTableLayoutPanel->Size = System::Drawing::Size(600, 200);
                this->LoginPrimaryTableLayoutPanel->TabIndex = 2;
                // 
                // CFormLogin
                // 
                this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
                this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
                this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                   static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->ClientSize = System::Drawing::Size(600, 200);
                this->Controls->Add(this->LoginPrimaryTableLayoutPanel);
                this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::None;
                this->MaximizeBox = false;
                this->MinimumSize = System::Drawing::Size(600, 200);
                this->Name = L"CFormLogin";
                this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
                this->Text = L"MyEA - Login";
                this->LoginSecondaryTableLayoutPanel->ResumeLayout(false);
                this->LoginTopTableLayoutPanel->ResumeLayout(false);
                this->LoginFormLayoutPanel->ResumeLayout(false);
                this->LoginFormLayoutPanel->PerformLayout();
                this->LoginPrimaryTableLayoutPanel->ResumeLayout(false);
                this->ResumeLayout(false);

            }
    #pragma endregion
        private: System::Void FormNN_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void FormMain_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void FormMain_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void LOGIN_BUTTON_CLOSE_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void LOGIN_BUTTON_SIGN_IN_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void LOGIN_BUTTON_SIGN_IN_KeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
};
    }
}
