#pragma once

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
        /// Summary for CFormAbout
        /// </summary>
        public ref class CFormAbout : public System::Windows::Forms::Form
        {
            private: System::Resources::ResourceManager^ _Ptr_Ressource_Manager;
        public:
            CFormAbout(System::Resources::ResourceManager^ ptr_rm_received);

            System::Void FormClose(System::Void);

        private:
            System::Boolean _moving;

            System::Drawing::Point _position_offset;

        protected:
            /// <summary>
            /// Clean up any resources being used.
            /// </summary>
            ~CFormAbout();

            private: System::Windows::Forms::TableLayoutPanel^  AboutSecondaryTableLayoutPanel;
            private: System::Windows::Forms::Label^  AboutCopyrightLabel;
            private: System::Windows::Forms::TableLayoutPanel^  AboutTopTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  AboutTitleTableLayoutPanel;
            private: System::Windows::Forms::Button^  ABOUTBUTTON_CLOSE;
            private: System::Windows::Forms::Label^  AboutTitleLabel;
            private: System::Windows::Forms::LinkLabel^  AboutLinkLabel;
            private: System::Windows::Forms::TableLayoutPanel^  AboutBottomTableLayoutPanel;
            private: System::Windows::Forms::Button^  ABOUTBUTTON_OK;
            private: System::Windows::Forms::Label^  ABOUTTITLE_LABEL;
            private: System::Windows::Forms::PictureBox^  ABOUTLOGO_PICTURE;
            private: System::Windows::Forms::TableLayoutPanel^  AboutPrimaryTableLayoutPanel;




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
                this->AboutSecondaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->AboutTopTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->ABOUTBUTTON_CLOSE = (gcnew System::Windows::Forms::Button());
                this->ABOUTTITLE_LABEL = (gcnew System::Windows::Forms::Label());
                this->AboutTitleTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->AboutTitleLabel = (gcnew System::Windows::Forms::Label());
                this->ABOUTLOGO_PICTURE = (gcnew System::Windows::Forms::PictureBox());
                this->AboutCopyrightLabel = (gcnew System::Windows::Forms::Label());
                this->AboutBottomTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->AboutLinkLabel = (gcnew System::Windows::Forms::LinkLabel());
                this->ABOUTBUTTON_OK = (gcnew System::Windows::Forms::Button());
                this->AboutPrimaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->AboutSecondaryTableLayoutPanel->SuspendLayout();
                this->AboutTopTableLayoutPanel->SuspendLayout();
                this->AboutTitleTableLayoutPanel->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ABOUTLOGO_PICTURE))->BeginInit();
                this->AboutBottomTableLayoutPanel->SuspendLayout();
                this->AboutPrimaryTableLayoutPanel->SuspendLayout();
                this->SuspendLayout();
                // 
                // AboutSecondaryTableLayoutPanel
                // 
                this->AboutSecondaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutSecondaryTableLayoutPanel->ColumnCount = 1;
                this->AboutSecondaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutSecondaryTableLayoutPanel->Controls->Add(this->AboutTopTableLayoutPanel, 0, 0);
                this->AboutSecondaryTableLayoutPanel->Controls->Add(this->AboutTitleTableLayoutPanel, 0, 1);
                this->AboutSecondaryTableLayoutPanel->Controls->Add(this->AboutCopyrightLabel, 0, 2);
                this->AboutSecondaryTableLayoutPanel->Controls->Add(this->AboutBottomTableLayoutPanel, 0, 3);
                this->AboutSecondaryTableLayoutPanel->Location = System::Drawing::Point(1, 1);
                this->AboutSecondaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->AboutSecondaryTableLayoutPanel->Name = "AboutSecondaryTableLayoutPanel";
                this->AboutSecondaryTableLayoutPanel->RowCount = 4;
                this->AboutSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->AboutSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    64)));
                this->AboutSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    50)));
                this->AboutSecondaryTableLayoutPanel->Size = System::Drawing::Size(598, 298);
                this->AboutSecondaryTableLayoutPanel->TabIndex = 1;
                // 
                // AboutTopTableLayoutPanel
                // 
                this->AboutTopTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutTopTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->AboutTopTableLayoutPanel->ColumnCount = 3;
                this->AboutTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    125)));
                this->AboutTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->AboutTopTableLayoutPanel->Controls->Add(this->ABOUTBUTTON_CLOSE, 2, 0);
                this->AboutTopTableLayoutPanel->Controls->Add(this->ABOUTTITLE_LABEL, 0, 0);
                this->AboutTopTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->AboutTopTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->AboutTopTableLayoutPanel->Name = "AboutTopTableLayoutPanel";
                this->AboutTopTableLayoutPanel->RowCount = 1;
                this->AboutTopTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutTopTableLayoutPanel->Size = System::Drawing::Size(598, 30);
                this->AboutTopTableLayoutPanel->TabIndex = 1;
                this->AboutTopTableLayoutPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormAbout::FormNN_Move_MouseDown);
                this->AboutTopTableLayoutPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormAbout::FormMain_Move_MouseMove);
                this->AboutTopTableLayoutPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormAbout::FormMain_Move_MouseUp);
                // 
                // ABOUTBUTTON_CLOSE
                // 
                this->ABOUTBUTTON_CLOSE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->ABOUTBUTTON_CLOSE->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->ABOUTBUTTON_CLOSE->FlatAppearance->BorderSize = 0;
                this->ABOUTBUTTON_CLOSE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ABOUTBUTTON_CLOSE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->ABOUTBUTTON_CLOSE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->ABOUTBUTTON_CLOSE->Location = System::Drawing::Point(568, 0);
                this->ABOUTBUTTON_CLOSE->Margin = System::Windows::Forms::Padding(0);
                this->ABOUTBUTTON_CLOSE->Name = "ABOUTBUTTON_CLOSE";
                this->ABOUTBUTTON_CLOSE->Size = System::Drawing::Size(30, 30);
                this->ABOUTBUTTON_CLOSE->TabIndex = 0;
                this->ABOUTBUTTON_CLOSE->UseVisualStyleBackColor = false;
                this->ABOUTBUTTON_CLOSE->Click += gcnew System::EventHandler(this, &CFormAbout::ABOUTBUTTON_CLOSE_Click);
                // 
                // ABOUTTITLE_LABEL
                // 
                this->ABOUTTITLE_LABEL->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->ABOUTTITLE_LABEL->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 9.75F));
                this->ABOUTTITLE_LABEL->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(153)),
                                                                                      static_cast<System::Int32>(static_cast<System::Byte>(153)), static_cast<System::Int32>(static_cast<System::Byte>(153)));
                this->ABOUTTITLE_LABEL->Location = System::Drawing::Point(0, 0);
                this->ABOUTTITLE_LABEL->Margin = System::Windows::Forms::Padding(0);
                this->ABOUTTITLE_LABEL->Name = "ABOUTTITLE_LABE";
                this->ABOUTTITLE_LABEL->Size = System::Drawing::Size(125, 30);
                this->ABOUTTITLE_LABEL->TabIndex = 1;
                this->ABOUTTITLE_LABEL->Text = "MyEA - About";
                this->ABOUTTITLE_LABEL->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                this->ABOUTTITLE_LABEL->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormAbout::FormNN_Move_MouseDown);
                this->ABOUTTITLE_LABEL->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormAbout::FormMain_Move_MouseMove);
                this->ABOUTTITLE_LABEL->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormAbout::FormMain_Move_MouseUp);
                // 
                // AboutTitleTableLayoutPanel
                // 
                this->AboutTitleTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutTitleTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                               static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->AboutTitleTableLayoutPanel->ColumnCount = 2;
                this->AboutTitleTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    64)));
                this->AboutTitleTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutTitleTableLayoutPanel->Controls->Add(this->AboutTitleLabel, 1, 0);
                this->AboutTitleTableLayoutPanel->Controls->Add(this->ABOUTLOGO_PICTURE, 0, 0);
                this->AboutTitleTableLayoutPanel->Location = System::Drawing::Point(0, 30);
                this->AboutTitleTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->AboutTitleTableLayoutPanel->Name = "AboutTitleTableLayoutPanel";
                this->AboutTitleTableLayoutPanel->RowCount = 1;
                this->AboutTitleTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutTitleTableLayoutPanel->Size = System::Drawing::Size(598, 64);
                this->AboutTitleTableLayoutPanel->TabIndex = 2;
                // 
                // AboutTitleLabel
                // 
                this->AboutTitleLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutTitleLabel->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 15.75F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->AboutTitleLabel->ForeColor = System::Drawing::Color::White;
                this->AboutTitleLabel->Location = System::Drawing::Point(64, 0);
                this->AboutTitleLabel->Margin = System::Windows::Forms::Padding(0);
                this->AboutTitleLabel->Name = "AboutTitleLabel";
                this->AboutTitleLabel->Size = System::Drawing::Size(534, 64);
                this->AboutTitleLabel->TabIndex = 0;
                this->AboutTitleLabel->Text = "MyEA";
                this->AboutTitleLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // ABOUTLOGO_PICTURE
                // 
                this->ABOUTLOGO_PICTURE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->ABOUTLOGO_PICTURE->Location = System::Drawing::Point(0, 0);
                this->ABOUTLOGO_PICTURE->Margin = System::Windows::Forms::Padding(0);
                this->ABOUTLOGO_PICTURE->Name = "ABOUTLOGO_PICTURE";
                this->ABOUTLOGO_PICTURE->Size = System::Drawing::Size(64, 64);
                this->ABOUTLOGO_PICTURE->TabIndex = 1;
                this->ABOUTLOGO_PICTURE->TabStop = false;
                this->ABOUTLOGO_PICTURE->Click += gcnew System::EventHandler(this, &CFormAbout::ABOUTLOGO_PICTURE_Click);
                // 
                // AboutCopyrightLabel
                // 
                this->AboutCopyrightLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutCopyrightLabel->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->AboutCopyrightLabel->ForeColor = System::Drawing::Color::White;
                this->AboutCopyrightLabel->Location = System::Drawing::Point(0, 94);
                this->AboutCopyrightLabel->Margin = System::Windows::Forms::Padding(0);
                this->AboutCopyrightLabel->Name = "AboutCopyrightLabel";
                this->AboutCopyrightLabel->Padding = System::Windows::Forms::Padding(5, 20, 0, 0);
                this->AboutCopyrightLabel->Size = System::Drawing::Size(598, 154);
                this->AboutCopyrightLabel->TabIndex = 0;
                this->AboutCopyrightLabel->Text = "MyEA\r\nVersion 0, 9, 0, 0\r\nCopyright 2016 Sébastien Kéroack.\r\nAll rights reserved."
                    "";
                // 
                // AboutBottomTableLayoutPanel
                // 
                this->AboutBottomTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutBottomTableLayoutPanel->ColumnCount = 2;
                this->AboutBottomTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutBottomTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    150)));
                this->AboutBottomTableLayoutPanel->Controls->Add(this->AboutLinkLabel, 0, 0);
                this->AboutBottomTableLayoutPanel->Controls->Add(this->ABOUTBUTTON_OK, 1, 0);
                this->AboutBottomTableLayoutPanel->Location = System::Drawing::Point(0, 248);
                this->AboutBottomTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->AboutBottomTableLayoutPanel->Name = "AboutBottomTableLayoutPanel";
                this->AboutBottomTableLayoutPanel->RowCount = 1;
                this->AboutBottomTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutBottomTableLayoutPanel->Size = System::Drawing::Size(598, 50);
                this->AboutBottomTableLayoutPanel->TabIndex = 3;
                // 
                // AboutLinkLabel
                // 
                this->AboutLinkLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutLinkLabel->LinkColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)), static_cast<System::Int32>(static_cast<System::Byte>(122)),
                                                                                   static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->AboutLinkLabel->Location = System::Drawing::Point(0, 0);
                this->AboutLinkLabel->Margin = System::Windows::Forms::Padding(0);
                this->AboutLinkLabel->Name = "AboutLinkLabel";
                this->AboutLinkLabel->Padding = System::Windows::Forms::Padding(5, 0, 0, 10);
                this->AboutLinkLabel->Size = System::Drawing::Size(448, 50);
                this->AboutLinkLabel->TabIndex = 3;
                this->AboutLinkLabel->TabStop = true;
                this->AboutLinkLabel->Text = "http://www.cataclysmique.net/";
                this->AboutLinkLabel->TextAlign = System::Drawing::ContentAlignment::BottomLeft;
                // 
                // ABOUTBUTTON_OK
                // 
                this->ABOUTBUTTON_OK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->ABOUTBUTTON_OK->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                    static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->ABOUTBUTTON_OK->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ABOUTBUTTON_OK->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->ABOUTBUTTON_OK->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->ABOUTBUTTON_OK->ForeColor = System::Drawing::Color::White;
                this->ABOUTBUTTON_OK->Location = System::Drawing::Point(458, 10);
                this->ABOUTBUTTON_OK->Margin = System::Windows::Forms::Padding(10);
                this->ABOUTBUTTON_OK->Name = "ABOUTBUTTON_OK";
                this->ABOUTBUTTON_OK->Size = System::Drawing::Size(130, 30);
                this->ABOUTBUTTON_OK->TabIndex = 4;
                this->ABOUTBUTTON_OK->Text = "OK";
                this->ABOUTBUTTON_OK->UseVisualStyleBackColor = false;
                this->ABOUTBUTTON_OK->Click += gcnew System::EventHandler(this, &CFormAbout::ABOUTBUTTON_OK_Click);
                // 
                // AboutPrimaryTableLayoutPanel
                // 
                this->AboutPrimaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AboutPrimaryTableLayoutPanel->ColumnCount = 3;
                this->AboutPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->AboutPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->AboutPrimaryTableLayoutPanel->Controls->Add(this->AboutSecondaryTableLayoutPanel, 1, 1);
                this->AboutPrimaryTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->AboutPrimaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->AboutPrimaryTableLayoutPanel->Name = "AboutPrimaryTableLayoutPanel";
                this->AboutPrimaryTableLayoutPanel->RowCount = 3;
                this->AboutPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->AboutPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AboutPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->AboutPrimaryTableLayoutPanel->Size = System::Drawing::Size(600, 300);
                this->AboutPrimaryTableLayoutPanel->TabIndex = 2;
                // 
                // CFormAbout
                // 
                this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
                this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
                this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                   static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->ClientSize = System::Drawing::Size(600, 300);
                this->Controls->Add(this->AboutPrimaryTableLayoutPanel);
                this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::None;
                this->MaximizeBox = false;
                this->MinimumSize = System::Drawing::Size(600, 300);
                this->Name = "CFormAbout";
                this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
                this->Text = "MyEA - About";
                this->AboutSecondaryTableLayoutPanel->ResumeLayout(false);
                this->AboutTopTableLayoutPanel->ResumeLayout(false);
                this->AboutTitleTableLayoutPanel->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ABOUTLOGO_PICTURE))->EndInit();
                this->AboutBottomTableLayoutPanel->ResumeLayout(false);
                this->AboutPrimaryTableLayoutPanel->ResumeLayout(false);
                this->ResumeLayout(false);

            }
    #pragma endregion
        private: System::Void FormNN_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void FormMain_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void FormMain_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void ABOUTBUTTON_CLOSE_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void ABOUTBUTTON_OK_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void ABOUTLOGO_PICTURE_Click(System::Object^  sender, System::EventArgs^  e) {
        }
};
    }
}
