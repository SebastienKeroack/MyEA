#pragma once

#include <vcclr.h>
#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

#include <Enums/Enum_Type_Indicator.hpp>
#include <Enums/Enum_Type_Main_Panel.hpp>
#include <Enums/Enum_Type_Position.hpp>
#include <Enums/Enum_Type_RichTextBox.hpp>
#include <Enums/Enum_Type_Signal_Pattern.hpp>
#include <Enums/Enum_Type_Signal_Quality.hpp>
#include <Enums/Enum_Type_TextBox.hpp>

#include <CMyEA.hpp>

#include <Form/CFormAbout.h>

namespace MyEA
{
    namespace Form
    {
        using namespace System;
        using namespace System::Resources;
        using namespace System::ComponentModel;
        using namespace System::Collections;
        using namespace System::Collections::Generic;
        using namespace System::Windows::Forms;
        using namespace System::Data;
        using namespace System::Drawing;

        /// <summary>
        /// Summary for CFormMain
        /// </summary>
        public ref class CFormMain : public System::Windows::Forms::Form
        {
            protected:
                ~CFormMain(System::Void);

                virtual System::Void OnPaint(PaintEventArgs^ e) override;
                virtual System::Void OnPaintBackground(PaintEventArgs^ e) override;

                unsigned int p_min_form_width;
                unsigned int p_min_form_height;

                MyEA::Common::ENUM_TYPE_MAIN_PANEL p_type_main_panel;

                List<System::Int32>^ p_Ptr_l_indicator;

                //Common::CMyEA* p_Ptr_CMyEA;

                CFormAbout^ p_Ptr_CFA;

                System::Resources::ResourceManager^ p_Ptr_Ressource_Manager;

            public:
                CFormMain(void);

                System::Void Exit(const System::Boolean exiTprocess_received);
                System::Void Cin_RichTextBox(const Common::ENUM_TYPE_RICHTEXTBOX tRichTextBox_received, System::String^ ptr_texTreceived, const System::UInt32 hex_color_received, const System::Boolean append_texTreceived, const System::Boolean new_line_received);
                System::Void Cin_TextBox(const Common::ENUM_TYPE_TEXTBOX tTextBox_received, System::String^ ptr_texTreceived);
                System::Void Cin_AccountInfo(System::String^ cin_texTreceived);
                System::Void Cin_PositionInfo(System::String^ cin_texTreceived);
                System::Void Signal_Modify(const System::Double condition_received,
                                                           const System::UInt32 filter_total_received,
                                                           const System::UInt32 filter_total_low_received,
                                                           const System::UInt32 filter_total_med_received,
                                                           const System::UInt32 filter_total_high_received,
                                                           System::String^ tool_tip_received,
                                                           Common::ENUM_EVENTSIGNAL_WAIT event_signal_waiTreceived);

                const System::Boolean FormLogin(void);
                const System::Boolean Init_CE(Common::CExpert* ptr_CE_received);
                const System::Boolean Signal_Filter_Ratio(const System::UInt16 type_received, 
                                                                                    const System::Double condition_received,
                                                                                    System::String^ symbol_received);
                const System::Boolean Signal_Filter_Push_Back(const Common::ENUM_TYPE_INDICATORS tIndicator_received,
                                                                                                System::String^ symbol_received,
                                                                                                System::String^ period_bars_received,
                                                                                                System::String^ tool_tip_received,
                                                                                                const Common::ENUM_TIME_FRAMES time_frames_received);
                const System::Boolean Signal_Filter_Modify(const Common::ENUM_TYPE_INDICATORS tIndicator_received,
                                                                                        const System::Int32 condition_received,
                                                                                        const System::Int32 pattern_received,
                                                                                        const Common::ENUM_TYPE_SIGNAL_PATTERN tSignal_Pattern_received,
                                                                                        const Common::ENUM_TYPE_SIGNAL_QUALITY tSignal_Quality_received,
                                                                                        System::String^ symbol_received,
                                                                                        System::String^ period_bars_received,
                                                                                        System::String^ tool_tip_received,
                                                                                        const Common::ENUM_TIME_FRAMES time_frames_received);
                const System::Boolean History_UnShift(const Common::ENUM_TYPE_POSITION tPosition_received,
                                                                                const System::UInt64 order_received,  
                                                                                const System::Double price_received,
                                                                                const System::Double stop_level_received,
                                                                                const System::Double take_level_received,
                                                                                System::String^ symbol_received,
                                                                                System::String^ date_0_received,
                                                                                System::String^ date_1_received,
                                                                                System::String^ comment_received,
                                                                                System::String^ tool_tip_received);
                System::Boolean isThreading;

                //Common::CMyEA* Get__Ptr_CMyEA(System::Void) { return(p_Ptr_CMyEA); }

            private:
                System::Void ReSize(const System::UInt16 id_received);
                System::Void Maximized(System::Void);
                System::Void Trade_GoTo_Received(const System::UInt16 type_received);
                System::Void Ratio_GoTo_Received(const System::UInt16 type_received);
                System::Void Main_GoTo_Received(Common::ENUM_TYPE_MAIN_PANEL tMainPanel_received);
                System::Void EnableTableLayoutPanel(TableLayoutPanel^ tlp_received, const System::Boolean enable_received);
                System::Void AutoResizeListView(ListView^ lisTview_received);
                System::Void AutoResizeSignalFilterListView(System::Void);

                const System::Boolean Main_GoTo_Init_(System::Void);
                System::Boolean _moving;
                System::Boolean _sizing;

                System::Drawing::Point _position_offset;
                System::Drawing::Point _size_offset;

            private: System::Windows::Forms::Label^  AccountInfo_Title_Label;
            private: System::Windows::Forms::Label^  AccountInfo_Desc_Label;
            private: System::Windows::Forms::Label^  PositionInfo_Title_Label;
            private: System::Windows::Forms::Label^  PositionInfo_Desc_Label;
            private: System::Windows::Forms::Button^  BUTTON_TRADE_RATIO;
            private: System::Windows::Forms::TableLayoutPanel^  TradeRatioTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  TradeTypeRatioTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  TradeRatioMenuTableLayoutPanel;
            private: System::Windows::Forms::Button^  BUTTON_RATIO_FLOOR;
            private: System::Windows::Forms::Button^  BUTTON_RATIO_ROUND;
            private: System::Windows::Forms::ListView^  TradeRatioFloorListView;
            private: System::Windows::Forms::ColumnHeader^  columnHeader48;
            private: System::Windows::Forms::ColumnHeader^  columnHeader49;
            private: System::Windows::Forms::ColumnHeader^  columnHeader50;
            private: System::Windows::Forms::ColumnHeader^  columnHeader51;
            private: System::Windows::Forms::ColumnHeader^  columnHeader52;
            private: System::Windows::Forms::ListView^  TradeRatioRoundListView;
            private: System::Windows::Forms::ColumnHeader^  columnHeader38;
            private: System::Windows::Forms::ColumnHeader^  columnHeader39;
            private: System::Windows::Forms::ColumnHeader^  columnHeader40;
            private: System::Windows::Forms::ColumnHeader^  columnHeader41;
            private: System::Windows::Forms::ColumnHeader^  columnHeader42;
            private: System::Windows::Forms::TableLayoutPanel^  TradeHistoryTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  TradeTradeTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  TradeOperationsTableLayoutPanel;
            private: System::Windows::Forms::Button^  BUTTON_FILE;
            private: System::Windows::Forms::Button^  BUTTON_VIEW;
            private: System::Windows::Forms::MenuStrip^  Help_MenuStrip;
            private: System::Windows::Forms::ToolStripMenuItem^  ToolStripMenuItem1;
            private: System::Windows::Forms::ToolStripMenuItem^  ViewHelpToolStripMenuItem;
            private: System::Windows::Forms::ToolStripSeparator^  ToolStripSeparator1;
            private: System::Windows::Forms::ToolStripMenuItem^  AboutMyEAToolStripMenuItem;
            private: System::Windows::Forms::ColumnHeader^  SignalColumnHeader5;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader1;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader2;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader6;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader7;
            private: System::Windows::Forms::ListView^  SignalListView;
            private: System::Windows::Forms::ColumnHeader^  SignalColumnHeader0;
            private: System::Windows::Forms::ColumnHeader^  SignalColumnHeader1;
            private: System::Windows::Forms::ColumnHeader^  SignalColumnHeader2;
            private: System::Windows::Forms::ColumnHeader^  SignalColumnHeader3;
            private: System::Windows::Forms::ColumnHeader^  SignalColumnHeader4;
            private: System::Windows::Forms::Button^  BUTTON_MENU_TRADE;
            private: System::Windows::Forms::TableLayoutPanel^  TradeTableLayoutPanel;
            private: System::Windows::Forms::ListView^  TradeHistoryListView;
            private: System::Windows::Forms::ColumnHeader^  columnHeader1;
            private: System::Windows::Forms::ColumnHeader^  columnHeader2;
            private: System::Windows::Forms::ColumnHeader^  columnHeader3;
            private: System::Windows::Forms::ColumnHeader^  columnHeader4;
            private: System::Windows::Forms::ColumnHeader^  columnHeader5;
            private: System::Windows::Forms::ColumnHeader^  columnHeader6;
            private: System::Windows::Forms::ColumnHeader^  columnHeader7;
            private: System::Windows::Forms::ColumnHeader^  columnHeader8;
            private: System::Windows::Forms::ColumnHeader^  columnHeader9;
            private: System::Windows::Forms::ColumnHeader^  columnHeader10;
            private: System::Windows::Forms::ColumnHeader^  columnHeader11;
            private: System::Windows::Forms::ListView^  TradeOperationsListView;
            private: System::Windows::Forms::TableLayoutPanel^  TradeMenuTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  OutputMenuTableLayoutPanel;
            private: System::Windows::Forms::Button^  BUTTON_TRADE_HISTORY;
            private: System::Windows::Forms::Button^  BUTTON_TRADE_OPERATIONS;
            private: System::Windows::Forms::TableLayoutPanel^  AccountTableLayoutPanel;
            private: System::Windows::Forms::Panel^  MainMenuBottomBorderPanel;
            private: System::Windows::Forms::TableLayoutPanel^  MainMenuButtonTableLayoutPanel;
            private: System::Windows::Forms::Button^  BUTTON_TRADE_TRADE;
            private: System::Windows::Forms::ListView^  TradeTradeListView;
            private: System::Windows::Forms::ColumnHeader^  columnHeader12;
            private: System::Windows::Forms::ColumnHeader^  columnHeader13;
            private: System::Windows::Forms::ColumnHeader^  columnHeader14;
            private: System::Windows::Forms::ColumnHeader^  columnHeader15;
            private: System::Windows::Forms::ColumnHeader^  columnHeader16;
            private: System::Windows::Forms::ColumnHeader^  columnHeader17;
            private: System::Windows::Forms::ColumnHeader^  columnHeader18;
            private: System::Windows::Forms::ColumnHeader^  columnHeader19;
            private: System::Windows::Forms::ColumnHeader^  columnHeader20;
            private: System::Windows::Forms::ColumnHeader^  columnHeader21;
            private: System::Windows::Forms::ColumnHeader^  columnHeader22;
            private: System::Windows::Forms::ColumnHeader^  columnHeader23;
            private: System::Windows::Forms::ColumnHeader^  columnHeader24;
            private: System::Windows::Forms::ColumnHeader^  columnHeader25;
            private: System::Windows::Forms::ColumnHeader^  columnHeader26;
            private: System::Windows::Forms::ColumnHeader^  columnHeader27;
            private: System::Windows::Forms::ColumnHeader^  columnHeader28;
            private: System::Windows::Forms::ColumnHeader^  columnHeader29;
            private: System::Windows::Forms::ColumnHeader^  columnHeader30;
            private: System::Windows::Forms::ColumnHeader^  columnHeader31;
            private: System::Windows::Forms::ColumnHeader^  columnHeader32;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader3;
            private: System::Windows::Forms::RichTextBox^  StatusPriceRichTextBox;
            private: System::Windows::Forms::RichTextBox^  StatusMoneyRichTextBox;
            private: System::Windows::Forms::RichTextBox^  StatusNetworkRichTextBox;
            private: System::Windows::Forms::TableLayoutPanel^  OutputTableLayoutPanel;
            private: System::Windows::Forms::RichTextBox^  OutputRichTextBox;
            private: System::Windows::Forms::TableLayoutPanel^  SignalTableLayoutPanel;
            private: System::Windows::Forms::ListView^  SignalFilterListView;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader0;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader4;
            private: System::Windows::Forms::ColumnHeader^  SignalFilterColumnHeader5;
            private: System::Windows::Forms::TableLayoutPanel^  FormMainPrimaryTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  FormMainSecondaryTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  MainMenuTableLayoutPanel;
            private: System::Windows::Forms::Panel^  BottomBorderPanel;
            private: System::Windows::Forms::Button^  BUTTON_MENU_OUTPUT;
            private: System::Windows::Forms::Button^  BUTTON_MENU_SIGNAL;
            private: System::Windows::Forms::Button^  BUTTON_MENU_ACCOUNT;
            private: System::Windows::Forms::PictureBox^  LOGO;
            private: System::Windows::Forms::Panel^  TopBorderPanel;
            private: System::Windows::Forms::RichTextBox^  StatusGeneralRichTextBox;
            private: System::Windows::Forms::Label^  FORM_TITLE_LABEL;
            private: System::Windows::Forms::Button^  BUTTON_CLOSE;
            private: System::Windows::Forms::Button^  BUTTON_MAXIMIZE;
            private: System::Windows::Forms::Button^  BUTTON_MINIMIZE;
            private: System::Windows::Forms::TableLayoutPanel^  FormTopTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  FormMenuTableLayoutPanel;
            private: System::Windows::Forms::Panel^  RightBorderPanel;
            private: System::Windows::Forms::TableLayoutPanel^  StatusTableLayoutPanel;
            private: System::Windows::Forms::Panel^  LeftBorderPanel;
            private: System::Windows::Forms::TableLayoutPanel^  MainTableLayoutPanel;

            private:
            /// <summary>
            /// Required designer variable.
            /// </summary>
            System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
            /// <summary>
            /// Required method for Designer support - do not modify
            /// the contents of this method with the code editor.
            /// </summary>
            void InitializeComponent(void)
            {
                System::Windows::Forms::ListViewItem^  listViewItem1 = (gcnew System::Windows::Forms::ListViewItem(gcnew cli::array< System::String^  >(6) {
                    "0",
                        "0", "0", "0", "0", "EVENTSIGNAL_WAIt_NONE"
                }, -1, System::Drawing::Color::White, System::Drawing::Color::Empty, nullptr));
                this->FORM_TITLE_LABEL = (gcnew System::Windows::Forms::Label());
                this->BUTTON_CLOSE = (gcnew System::Windows::Forms::Button());
                this->BUTTON_MAXIMIZE = (gcnew System::Windows::Forms::Button());
                this->BUTTON_MINIMIZE = (gcnew System::Windows::Forms::Button());
                this->FormTopTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->LOGO = (gcnew System::Windows::Forms::PictureBox());
                this->FormMenuTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->BUTTON_FILE = (gcnew System::Windows::Forms::Button());
                this->BUTTON_VIEW = (gcnew System::Windows::Forms::Button());
                this->Help_MenuStrip = (gcnew System::Windows::Forms::MenuStrip());
                this->ToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
                this->ViewHelpToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
                this->ToolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
                this->AboutMyEAToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
                this->RightBorderPanel = (gcnew System::Windows::Forms::Panel());
                this->StatusTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->StatusGeneralRichTextBox = (gcnew System::Windows::Forms::RichTextBox());
                this->StatusPriceRichTextBox = (gcnew System::Windows::Forms::RichTextBox());
                this->StatusMoneyRichTextBox = (gcnew System::Windows::Forms::RichTextBox());
                this->StatusNetworkRichTextBox = (gcnew System::Windows::Forms::RichTextBox());
                this->LeftBorderPanel = (gcnew System::Windows::Forms::Panel());
                this->MainTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->MainMenuTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->MainMenuBottomBorderPanel = (gcnew System::Windows::Forms::Panel());
                this->MainMenuButtonTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->BUTTON_MENU_OUTPUT = (gcnew System::Windows::Forms::Button());
                this->BUTTON_MENU_SIGNAL = (gcnew System::Windows::Forms::Button());
                this->BUTTON_MENU_ACCOUNT = (gcnew System::Windows::Forms::Button());
                this->BUTTON_MENU_TRADE = (gcnew System::Windows::Forms::Button());
                this->OutputTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->OutputRichTextBox = (gcnew System::Windows::Forms::RichTextBox());
                this->OutputMenuTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->SignalTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->SignalFilterListView = (gcnew System::Windows::Forms::ListView());
                this->SignalFilterColumnHeader0 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader1 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader2 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader3 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader4 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader5 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader6 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalFilterColumnHeader7 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalListView = (gcnew System::Windows::Forms::ListView());
                this->SignalColumnHeader0 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalColumnHeader1 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalColumnHeader2 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalColumnHeader3 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalColumnHeader4 = (gcnew System::Windows::Forms::ColumnHeader());
                this->SignalColumnHeader5 = (gcnew System::Windows::Forms::ColumnHeader());
                this->AccountTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->AccountInfo_Title_Label = (gcnew System::Windows::Forms::Label());
                this->AccountInfo_Desc_Label = (gcnew System::Windows::Forms::Label());
                this->PositionInfo_Title_Label = (gcnew System::Windows::Forms::Label());
                this->PositionInfo_Desc_Label = (gcnew System::Windows::Forms::Label());
                this->TradeTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->TradeMenuTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->BUTTON_TRADE_HISTORY = (gcnew System::Windows::Forms::Button());
                this->BUTTON_TRADE_TRADE = (gcnew System::Windows::Forms::Button());
                this->BUTTON_TRADE_RATIO = (gcnew System::Windows::Forms::Button());
                this->BUTTON_TRADE_OPERATIONS = (gcnew System::Windows::Forms::Button());
                this->TradeRatioTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->TradeTypeRatioTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->TradeRatioFloorListView = (gcnew System::Windows::Forms::ListView());
                this->columnHeader48 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader49 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader50 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader51 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader52 = (gcnew System::Windows::Forms::ColumnHeader());
                this->TradeRatioRoundListView = (gcnew System::Windows::Forms::ListView());
                this->columnHeader38 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader39 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader40 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader41 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader42 = (gcnew System::Windows::Forms::ColumnHeader());
                this->TradeRatioMenuTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->BUTTON_RATIO_FLOOR = (gcnew System::Windows::Forms::Button());
                this->BUTTON_RATIO_ROUND = (gcnew System::Windows::Forms::Button());
                this->TradeHistoryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->TradeHistoryListView = (gcnew System::Windows::Forms::ListView());
                this->columnHeader1 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader2 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader3 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader4 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader5 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader6 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader7 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader8 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader9 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader10 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader11 = (gcnew System::Windows::Forms::ColumnHeader());
                this->TradeTradeTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->TradeTradeListView = (gcnew System::Windows::Forms::ListView());
                this->columnHeader22 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader23 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader24 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader25 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader26 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader27 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader28 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader29 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader30 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader31 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader32 = (gcnew System::Windows::Forms::ColumnHeader());
                this->TradeOperationsTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->TradeOperationsListView = (gcnew System::Windows::Forms::ListView());
                this->columnHeader12 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader13 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader14 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader15 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader16 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader17 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader18 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader19 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader20 = (gcnew System::Windows::Forms::ColumnHeader());
                this->columnHeader21 = (gcnew System::Windows::Forms::ColumnHeader());
                this->TopBorderPanel = (gcnew System::Windows::Forms::Panel());
                this->FormMainPrimaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->FormMainSecondaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->BottomBorderPanel = (gcnew System::Windows::Forms::Panel());
                this->FormTopTableLayoutPanel->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->LOGO))->BeginInit();
                this->FormMenuTableLayoutPanel->SuspendLayout();
                this->Help_MenuStrip->SuspendLayout();
                this->StatusTableLayoutPanel->SuspendLayout();
                this->MainTableLayoutPanel->SuspendLayout();
                this->MainMenuTableLayoutPanel->SuspendLayout();
                this->MainMenuButtonTableLayoutPanel->SuspendLayout();
                this->OutputTableLayoutPanel->SuspendLayout();
                this->SignalTableLayoutPanel->SuspendLayout();
                this->AccountTableLayoutPanel->SuspendLayout();
                this->TradeTableLayoutPanel->SuspendLayout();
                this->TradeMenuTableLayoutPanel->SuspendLayout();
                this->TradeRatioTableLayoutPanel->SuspendLayout();
                this->TradeTypeRatioTableLayoutPanel->SuspendLayout();
                this->TradeRatioMenuTableLayoutPanel->SuspendLayout();
                this->TradeHistoryTableLayoutPanel->SuspendLayout();
                this->TradeTradeTableLayoutPanel->SuspendLayout();
                this->TradeOperationsTableLayoutPanel->SuspendLayout();
                this->FormMainPrimaryTableLayoutPanel->SuspendLayout();
                this->FormMainSecondaryTableLayoutPanel->SuspendLayout();
                this->SuspendLayout();
                // 
                // FORM_TITLE_LABEL
                // 
                this->FORM_TITLE_LABEL->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->FORM_TITLE_LABEL->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                     static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->FORM_TITLE_LABEL->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->FORM_TITLE_LABEL->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(153)),
                                                                                     static_cast<System::Int32>(static_cast<System::Byte>(153)), static_cast<System::Int32>(static_cast<System::Byte>(153)));
                this->FORM_TITLE_LABEL->Location = System::Drawing::Point(33, 0);
                this->FORM_TITLE_LABEL->Margin = System::Windows::Forms::Padding(0);
                this->FORM_TITLE_LABEL->Name = "FORM_TITLE_LABE";
                this->FORM_TITLE_LABEL->Size = System::Drawing::Size(515, 30);
                this->FORM_TITLE_LABEL->TabIndex = 1;
                this->FORM_TITLE_LABEL->Text = "MyEA";
                this->FORM_TITLE_LABEL->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                this->FORM_TITLE_LABEL->DoubleClick += gcnew System::EventHandler(this, &CFormMain::Maximized_DoubleClick);
                this->FORM_TITLE_LABEL->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseDown);
                this->FORM_TITLE_LABEL->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseMove);
                this->FORM_TITLE_LABEL->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseUp);
                // 
                // BUTTON_CLOSE
                // 
                this->BUTTON_CLOSE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_CLOSE->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->BUTTON_CLOSE->FlatAppearance->BorderSize = 0;
                this->BUTTON_CLOSE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                          static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_CLOSE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                          static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->BUTTON_CLOSE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_CLOSE->Location = System::Drawing::Point(608, 0);
                this->BUTTON_CLOSE->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_CLOSE->Name = "BUTTON_CLOSE";
                this->BUTTON_CLOSE->Size = System::Drawing::Size(30, 30);
                this->BUTTON_CLOSE->TabIndex = 2;
                this->BUTTON_CLOSE->UseVisualStyleBackColor = false;
                this->BUTTON_CLOSE->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_CLOSE_Click);
                // 
                // BUTTON_MAXIMIZE
                // 
                this->BUTTON_MAXIMIZE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_MAXIMIZE->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->BUTTON_MAXIMIZE->FlatAppearance->BorderSize = 0;
                this->BUTTON_MAXIMIZE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MAXIMIZE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->BUTTON_MAXIMIZE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_MAXIMIZE->Location = System::Drawing::Point(578, 0);
                this->BUTTON_MAXIMIZE->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_MAXIMIZE->Name = "BUTTON_MAXIMIZE";
                this->BUTTON_MAXIMIZE->Size = System::Drawing::Size(30, 30);
                this->BUTTON_MAXIMIZE->TabIndex = 3;
                this->BUTTON_MAXIMIZE->UseVisualStyleBackColor = false;
                this->BUTTON_MAXIMIZE->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_MAXIMIZE_Click);
                // 
                // BUTTON_MINIMIZE
                // 
                this->BUTTON_MINIMIZE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_MINIMIZE->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->BUTTON_MINIMIZE->FlatAppearance->BorderSize = 0;
                this->BUTTON_MINIMIZE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MINIMIZE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->BUTTON_MINIMIZE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_MINIMIZE->Location = System::Drawing::Point(548, 0);
                this->BUTTON_MINIMIZE->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_MINIMIZE->Name = "BUTTON_MINIMIZE";
                this->BUTTON_MINIMIZE->Size = System::Drawing::Size(30, 30);
                this->BUTTON_MINIMIZE->TabIndex = 4;
                this->BUTTON_MINIMIZE->UseVisualStyleBackColor = false;
                this->BUTTON_MINIMIZE->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_MINIMIZE_Click);
                // 
                // FormTopTableLayoutPanel
                // 
                this->FormTopTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->FormTopTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                            static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->FormTopTableLayoutPanel->ColumnCount = 5;
                this->FormTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->FormTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->FormTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->FormTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->FormTopTableLayoutPanel->Controls->Add(this->LOGO, 0, 0);
                this->FormTopTableLayoutPanel->Controls->Add(this->FORM_TITLE_LABEL, 1, 0);
                this->FormTopTableLayoutPanel->Controls->Add(this->BUTTON_MINIMIZE, 2, 0);
                this->FormTopTableLayoutPanel->Controls->Add(this->BUTTON_MAXIMIZE, 3, 0);
                this->FormTopTableLayoutPanel->Controls->Add(this->BUTTON_CLOSE, 4, 0);
                this->FormTopTableLayoutPanel->GrowStyle = System::Windows::Forms::TableLayoutPanelGrowStyle::AddColumns;
                this->FormTopTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->FormTopTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->FormTopTableLayoutPanel->Name = "FormTopTableLayoutPanel";
                this->FormTopTableLayoutPanel->RowCount = 1;
                this->FormTopTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormTopTableLayoutPanel->Size = System::Drawing::Size(638, 30);
                this->FormTopTableLayoutPanel->TabIndex = 0;
                this->FormTopTableLayoutPanel->DoubleClick += gcnew System::EventHandler(this, &CFormMain::Maximized_DoubleClick);
                this->FormTopTableLayoutPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseDown);
                this->FormTopTableLayoutPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseMove);
                this->FormTopTableLayoutPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseUp);
                // 
                // LOGO
                // 
                this->LOGO->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LOGO->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->LOGO->Location = System::Drawing::Point(3, 3);
                this->LOGO->Margin = System::Windows::Forms::Padding(3, 3, 0, 0);
                this->LOGO->Name = "LOGO";
                this->LOGO->Size = System::Drawing::Size(30, 27);
                this->LOGO->TabIndex = 1;
                this->LOGO->TabStop = false;
                this->LOGO->DoubleClick += gcnew System::EventHandler(this, &CFormMain::Maximized_DoubleClick);
                this->LOGO->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseDown);
                this->LOGO->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseMove);
                this->LOGO->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_Move_MouseUp);
                // 
                // FormMenuTableLayoutPanel
                // 
                this->FormMenuTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->FormMenuTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->FormMenuTableLayoutPanel->ColumnCount = 5;
                this->FormMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->FormMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->FormMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->FormMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    50)));
                this->FormMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormMenuTableLayoutPanel->Controls->Add(this->BUTTON_FILE, 0, 0);
                this->FormMenuTableLayoutPanel->Controls->Add(this->BUTTON_VIEW, 1, 0);
                this->FormMenuTableLayoutPanel->Controls->Add(this->Help_MenuStrip, 2, 0);
                this->FormMenuTableLayoutPanel->Location = System::Drawing::Point(5, 33);
                this->FormMenuTableLayoutPanel->Margin = System::Windows::Forms::Padding(5, 3, 5, 0);
                this->FormMenuTableLayoutPanel->Name = "FormMenuTableLayoutPanel";
                this->FormMenuTableLayoutPanel->RowCount = 1;
                this->FormMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->FormMenuTableLayoutPanel->Size = System::Drawing::Size(628, 25);
                this->FormMenuTableLayoutPanel->TabIndex = 5;
                // 
                // BUTTON_FILE
                // 
                this->BUTTON_FILE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left));
                this->BUTTON_FILE->FlatAppearance->BorderSize = 0;
                this->BUTTON_FILE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(27)),
                                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(27)), static_cast<System::Int32>(static_cast<System::Byte>(28)));
                this->BUTTON_FILE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(64)));
                this->BUTTON_FILE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_FILE->ForeColor = System::Drawing::Color::White;
                this->BUTTON_FILE->Location = System::Drawing::Point(0, 0);
                this->BUTTON_FILE->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_FILE->Name = "BUTTON_FILE";
                this->BUTTON_FILE->Size = System::Drawing::Size(50, 25);
                this->BUTTON_FILE->TabIndex = 6;
                this->BUTTON_FILE->Text = "EXIT";
                this->BUTTON_FILE->UseVisualStyleBackColor = true;
                this->BUTTON_FILE->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_FILE_Click);
                // 
                // BUTTON_VIEW
                // 
                this->BUTTON_VIEW->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left));
                this->BUTTON_VIEW->FlatAppearance->BorderSize = 0;
                this->BUTTON_VIEW->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(27)),
                                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(27)), static_cast<System::Int32>(static_cast<System::Byte>(28)));
                this->BUTTON_VIEW->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(64)));
                this->BUTTON_VIEW->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_VIEW->ForeColor = System::Drawing::Color::White;
                this->BUTTON_VIEW->Location = System::Drawing::Point(50, 0);
                this->BUTTON_VIEW->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_VIEW->Name = "BUTTON_VIEW";
                this->BUTTON_VIEW->Size = System::Drawing::Size(50, 25);
                this->BUTTON_VIEW->TabIndex = 7;
                this->BUTTON_VIEW->Text = "VIEW";
                this->BUTTON_VIEW->UseVisualStyleBackColor = true;
                this->BUTTON_VIEW->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_VIEW_Click);
                // 
                // Help_MenuStrip
                // 
                this->Help_MenuStrip->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->Help_MenuStrip->AutoSize = false;
                this->Help_MenuStrip->Dock = System::Windows::Forms::DockStyle::None;
                this->Help_MenuStrip->GripMargin = System::Windows::Forms::Padding(0);
                this->Help_MenuStrip->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->ToolStripMenuItem1 });
                this->Help_MenuStrip->LayoutStyle = System::Windows::Forms::ToolStripLayoutStyle::VerticalStackWithOverflow;
                this->Help_MenuStrip->Location = System::Drawing::Point(100, 0);
                this->Help_MenuStrip->Name = "Help_MenuStrip";
                this->Help_MenuStrip->Padding = System::Windows::Forms::Padding(0, 2, 0, 0);
                this->Help_MenuStrip->Size = System::Drawing::Size(50, 25);
                this->Help_MenuStrip->TabIndex = 9;
                // 
                // ToolStripMenuItem1
                // 
                this->ToolStripMenuItem1->AutoSize = false;
                this->ToolStripMenuItem1->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Text;
                this->ToolStripMenuItem1->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
                    this->ViewHelpToolStripMenuItem,
                        this->ToolStripSeparator1, this->AboutMyEAToolStripMenuItem
                });
                this->ToolStripMenuItem1->Name = "ToolStripMenuItem1";
                this->ToolStripMenuItem1->Padding = System::Windows::Forms::Padding(0);
                this->ToolStripMenuItem1->Size = System::Drawing::Size(50, 22);
                this->ToolStripMenuItem1->Text = "HELP";
                // 
                // ViewHelpToolStripMenuItem
                // 
                this->ViewHelpToolStripMenuItem->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                              static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->ViewHelpToolStripMenuItem->ForeColor = System::Drawing::Color::White;
                this->ViewHelpToolStripMenuItem->Name = "ViewHelpToolStripMenuItem";
                this->ViewHelpToolStripMenuItem->Size = System::Drawing::Size(141, 22);
                this->ViewHelpToolStripMenuItem->Text = "View Help";
                this->ViewHelpToolStripMenuItem->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // ToolStripSeparator1
                // 
                this->ToolStripSeparator1->AutoSize = false;
                this->ToolStripSeparator1->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                        static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->ToolStripSeparator1->Name = "ToolStripSeparator1";
                this->ToolStripSeparator1->Size = System::Drawing::Size(138, 5);
                // 
                // AboutMyEAToolStripMenuItem
                // 
                this->AboutMyEAToolStripMenuItem->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                               static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->AboutMyEAToolStripMenuItem->ForeColor = System::Drawing::Color::White;
                this->AboutMyEAToolStripMenuItem->Name = "AboutMyEAToolStripMenuItem";
                this->AboutMyEAToolStripMenuItem->Size = System::Drawing::Size(141, 22);
                this->AboutMyEAToolStripMenuItem->Text = "About MyEA";
                this->AboutMyEAToolStripMenuItem->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                this->AboutMyEAToolStripMenuItem->Click += gcnew System::EventHandler(this, &CFormMain::AboutMyEAToolStripMenuItem_Click);
                // 
                // RightBorderPanel
                // 
                this->RightBorderPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->RightBorderPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)), static_cast<System::Int32>(static_cast<System::Byte>(122)),
                                                                                     static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->RightBorderPanel->Cursor = System::Windows::Forms::Cursors::SizeWE;
                this->RightBorderPanel->Location = System::Drawing::Point(639, 1);
                this->RightBorderPanel->Margin = System::Windows::Forms::Padding(0);
                this->RightBorderPanel->Name = "RightBorderPanel";
                this->RightBorderPanel->Size = System::Drawing::Size(1, 478);
                this->RightBorderPanel->TabIndex = 0;
                this->RightBorderPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseDown);
                this->RightBorderPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseMove_Right);
                this->RightBorderPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseUp);
                // 
                // StatusTableLayoutPanel
                // 
                this->StatusTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->StatusTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                           static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->StatusTableLayoutPanel->ColumnCount = 4;
                this->StatusTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->StatusTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->StatusTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->StatusTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->StatusTableLayoutPanel->Controls->Add(this->StatusGeneralRichTextBox);
                this->StatusTableLayoutPanel->Controls->Add(this->StatusPriceRichTextBox, 1, 0);
                this->StatusTableLayoutPanel->Controls->Add(this->StatusMoneyRichTextBox, 2, 0);
                this->StatusTableLayoutPanel->Controls->Add(this->StatusNetworkRichTextBox, 3, 0);
                this->StatusTableLayoutPanel->GrowStyle = System::Windows::Forms::TableLayoutPanelGrowStyle::AddColumns;
                this->StatusTableLayoutPanel->Location = System::Drawing::Point(0, 453);
                this->StatusTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->StatusTableLayoutPanel->Name = "StatusTableLayoutPanel";
                this->StatusTableLayoutPanel->RowCount = 1;
                this->StatusTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->StatusTableLayoutPanel->Size = System::Drawing::Size(638, 25);
                this->StatusTableLayoutPanel->TabIndex = 4;
                // 
                // StatusGeneralRichTextBox
                // 
                this->StatusGeneralRichTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->StatusGeneralRichTextBox->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->StatusGeneralRichTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->StatusGeneralRichTextBox->Cursor = System::Windows::Forms::Cursors::Default;
                this->StatusGeneralRichTextBox->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->StatusGeneralRichTextBox->ForeColor = System::Drawing::Color::White;
                this->StatusGeneralRichTextBox->HideSelection = false;
                this->StatusGeneralRichTextBox->Location = System::Drawing::Point(3, 3);
                this->StatusGeneralRichTextBox->MaxLength = 32767;
                this->StatusGeneralRichTextBox->Multiline = false;
                this->StatusGeneralRichTextBox->Name = "StatusGeneralRichTextBox";
                this->StatusGeneralRichTextBox->ReadOnly = true;
                this->StatusGeneralRichTextBox->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->StatusGeneralRichTextBox->Size = System::Drawing::Size(164, 19);
                this->StatusGeneralRichTextBox->TabIndex = 0;
                this->StatusGeneralRichTextBox->TabStop = false;
                this->StatusGeneralRichTextBox->Text = "Status";
                this->StatusGeneralRichTextBox->WordWrap = false;
                // 
                // StatusPriceRichTextBox
                // 
                this->StatusPriceRichTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->StatusPriceRichTextBox->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                           static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->StatusPriceRichTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->StatusPriceRichTextBox->Cursor = System::Windows::Forms::Cursors::Default;
                this->StatusPriceRichTextBox->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->StatusPriceRichTextBox->ForeColor = System::Drawing::Color::White;
                this->StatusPriceRichTextBox->Location = System::Drawing::Point(173, 3);
                this->StatusPriceRichTextBox->MaxLength = 32767;
                this->StatusPriceRichTextBox->Multiline = false;
                this->StatusPriceRichTextBox->Name = "StatusPriceRichTextBox";
                this->StatusPriceRichTextBox->ReadOnly = true;
                this->StatusPriceRichTextBox->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->StatusPriceRichTextBox->Size = System::Drawing::Size(225, 19);
                this->StatusPriceRichTextBox->TabIndex = 1;
                this->StatusPriceRichTextBox->TabStop = false;
                this->StatusPriceRichTextBox->Text = "";
                this->StatusPriceRichTextBox->WordWrap = false;
                // 
                // StatusMoneyRichTextBox
                // 
                this->StatusMoneyRichTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->StatusMoneyRichTextBox->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                           static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->StatusMoneyRichTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->StatusMoneyRichTextBox->Cursor = System::Windows::Forms::Cursors::Default;
                this->StatusMoneyRichTextBox->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->StatusMoneyRichTextBox->ForeColor = System::Drawing::Color::White;
                this->StatusMoneyRichTextBox->Location = System::Drawing::Point(404, 3);
                this->StatusMoneyRichTextBox->MaxLength = 32767;
                this->StatusMoneyRichTextBox->Multiline = false;
                this->StatusMoneyRichTextBox->Name = "StatusMoneyRichTextBox";
                this->StatusMoneyRichTextBox->ReadOnly = true;
                this->StatusMoneyRichTextBox->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->StatusMoneyRichTextBox->Size = System::Drawing::Size(150, 19);
                this->StatusMoneyRichTextBox->TabIndex = 2;
                this->StatusMoneyRichTextBox->TabStop = false;
                this->StatusMoneyRichTextBox->Text = "";
                this->StatusMoneyRichTextBox->WordWrap = false;
                // 
                // StatusNetworkRichTextBox
                // 
                this->StatusNetworkRichTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->StatusNetworkRichTextBox->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->StatusNetworkRichTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->StatusNetworkRichTextBox->Cursor = System::Windows::Forms::Cursors::Default;
                this->StatusNetworkRichTextBox->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->StatusNetworkRichTextBox->ForeColor = System::Drawing::Color::White;
                this->StatusNetworkRichTextBox->Location = System::Drawing::Point(560, 3);
                this->StatusNetworkRichTextBox->MaxLength = 255;
                this->StatusNetworkRichTextBox->Multiline = false;
                this->StatusNetworkRichTextBox->Name = "StatusNetworkRichTextBox";
                this->StatusNetworkRichTextBox->ReadOnly = true;
                this->StatusNetworkRichTextBox->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->StatusNetworkRichTextBox->Size = System::Drawing::Size(75, 19);
                this->StatusNetworkRichTextBox->TabIndex = 3;
                this->StatusNetworkRichTextBox->TabStop = false;
                this->StatusNetworkRichTextBox->Text = "OFFLINE";
                this->StatusNetworkRichTextBox->WordWrap = false;
                // 
                // LeftBorderPanel
                // 
                this->LeftBorderPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->LeftBorderPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)), static_cast<System::Int32>(static_cast<System::Byte>(122)),
                                                                                    static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->LeftBorderPanel->Cursor = System::Windows::Forms::Cursors::SizeWE;
                this->LeftBorderPanel->Location = System::Drawing::Point(0, 1);
                this->LeftBorderPanel->Margin = System::Windows::Forms::Padding(0);
                this->LeftBorderPanel->Name = "LeftBorderPanel";
                this->LeftBorderPanel->Size = System::Drawing::Size(1, 478);
                this->LeftBorderPanel->TabIndex = 0;
                this->LeftBorderPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseDown);
                this->LeftBorderPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseMove_Left);
                this->LeftBorderPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseUp);
                // 
                // MainTableLayoutPanel
                // 
                this->MainTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MainTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->MainTableLayoutPanel->ColumnCount = 1;
                this->MainTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MainTableLayoutPanel->Controls->Add(this->MainMenuTableLayoutPanel, 0, 0);
                this->MainTableLayoutPanel->Controls->Add(this->OutputTableLayoutPanel, 0, 1);
                this->MainTableLayoutPanel->Controls->Add(this->SignalTableLayoutPanel, 0, 2);
                this->MainTableLayoutPanel->Controls->Add(this->AccountTableLayoutPanel, 0, 3);
                this->MainTableLayoutPanel->Controls->Add(this->TradeTableLayoutPanel, 0, 4);
                this->MainTableLayoutPanel->Location = System::Drawing::Point(5, 63);
                this->MainTableLayoutPanel->Margin = System::Windows::Forms::Padding(5);
                this->MainTableLayoutPanel->Name = "MainTableLayoutPanel";
                this->MainTableLayoutPanel->RowCount = 5;
                this->MainTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->MainTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->MainTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->MainTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->MainTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MainTableLayoutPanel->Size = System::Drawing::Size(628, 385);
                this->MainTableLayoutPanel->TabIndex = 5;
                // 
                // MainMenuTableLayoutPanel
                // 
                this->MainMenuTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MainMenuTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                             static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->MainMenuTableLayoutPanel->ColumnCount = 5;
                this->MainMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MainMenuTableLayoutPanel->Controls->Add(this->MainMenuBottomBorderPanel, 4, 1);
                this->MainMenuTableLayoutPanel->Controls->Add(this->MainMenuButtonTableLayoutPanel, 4, 0);
                this->MainMenuTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->MainMenuTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->MainMenuTableLayoutPanel->Name = "MainMenuTableLayoutPanel";
                this->MainMenuTableLayoutPanel->RowCount = 2;
                this->MainMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MainMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    2)));
                this->MainMenuTableLayoutPanel->Size = System::Drawing::Size(628, 25);
                this->MainMenuTableLayoutPanel->TabIndex = 11;
                // 
                // MainMenuBottomBorderPanel
                // 
                this->MainMenuBottomBorderPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MainMenuBottomBorderPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                              static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->MainMenuBottomBorderPanel->Location = System::Drawing::Point(0, 23);
                this->MainMenuBottomBorderPanel->Margin = System::Windows::Forms::Padding(0);
                this->MainMenuBottomBorderPanel->Name = "MainMenuBottomBorderPanel";
                this->MainMenuBottomBorderPanel->Size = System::Drawing::Size(628, 2);
                this->MainMenuBottomBorderPanel->TabIndex = 4;
                // 
                // MainMenuButtonTableLayoutPanel
                // 
                this->MainMenuButtonTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MainMenuButtonTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                                   static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->MainMenuButtonTableLayoutPanel->ColumnCount = 5;
                this->MainMenuButtonTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuButtonTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuButtonTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuButtonTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
                this->MainMenuButtonTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MainMenuButtonTableLayoutPanel->Controls->Add(this->BUTTON_MENU_OUTPUT, 0, 0);
                this->MainMenuButtonTableLayoutPanel->Controls->Add(this->BUTTON_MENU_SIGNAL, 1, 0);
                this->MainMenuButtonTableLayoutPanel->Controls->Add(this->BUTTON_MENU_ACCOUNT, 2, 0);
                this->MainMenuButtonTableLayoutPanel->Controls->Add(this->BUTTON_MENU_TRADE, 3, 0);
                this->MainMenuButtonTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->MainMenuButtonTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->MainMenuButtonTableLayoutPanel->Name = "MainMenuButtonTableLayoutPanel";
                this->MainMenuButtonTableLayoutPanel->RowCount = 1;
                this->MainMenuButtonTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MainMenuButtonTableLayoutPanel->Size = System::Drawing::Size(628, 23);
                this->MainMenuButtonTableLayoutPanel->TabIndex = 5;
                // 
                // BUTTON_MENU_OUTPUT
                // 
                this->BUTTON_MENU_OUTPUT->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_MENU_OUTPUT->FlatAppearance->BorderSize = 0;
                this->BUTTON_MENU_OUTPUT->FlatAppearance->CheckedBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                              static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_OUTPUT->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_OUTPUT->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_MENU_OUTPUT->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_MENU_OUTPUT->ForeColor = System::Drawing::Color::White;
                this->BUTTON_MENU_OUTPUT->Location = System::Drawing::Point(0, 0);
                this->BUTTON_MENU_OUTPUT->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_MENU_OUTPUT->Name = "BUTTON_MENU_OUTPUT";
                this->BUTTON_MENU_OUTPUT->Size = System::Drawing::Size(75, 23);
                this->BUTTON_MENU_OUTPUT->TabIndex = 0;
                this->BUTTON_MENU_OUTPUT->Text = "Output";
                this->BUTTON_MENU_OUTPUT->UseVisualStyleBackColor = true;
                this->BUTTON_MENU_OUTPUT->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_MENU_OUTPUTClick);
                // 
                // BUTTON_MENU_SIGNAL
                // 
                this->BUTTON_MENU_SIGNAL->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_MENU_SIGNAL->FlatAppearance->BorderSize = 0;
                this->BUTTON_MENU_SIGNAL->FlatAppearance->CheckedBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                              static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_SIGNAL->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_SIGNAL->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_MENU_SIGNAL->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_MENU_SIGNAL->ForeColor = System::Drawing::Color::White;
                this->BUTTON_MENU_SIGNAL->Location = System::Drawing::Point(75, 0);
                this->BUTTON_MENU_SIGNAL->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_MENU_SIGNAL->Name = "BUTTON_MENU_SIGNA";
                this->BUTTON_MENU_SIGNAL->Size = System::Drawing::Size(75, 23);
                this->BUTTON_MENU_SIGNAL->TabIndex = 1;
                this->BUTTON_MENU_SIGNAL->Text = "Signal";
                this->BUTTON_MENU_SIGNAL->UseVisualStyleBackColor = true;
                this->BUTTON_MENU_SIGNAL->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_MENU_SIGNAL_Click);
                // 
                // BUTTON_MENU_ACCOUNT
                // 
                this->BUTTON_MENU_ACCOUNT->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_MENU_ACCOUNT->FlatAppearance->BorderSize = 0;
                this->BUTTON_MENU_ACCOUNT->FlatAppearance->CheckedBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                               static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_ACCOUNT->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                 static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_ACCOUNT->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                 static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_MENU_ACCOUNT->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_MENU_ACCOUNT->ForeColor = System::Drawing::Color::White;
                this->BUTTON_MENU_ACCOUNT->Location = System::Drawing::Point(150, 0);
                this->BUTTON_MENU_ACCOUNT->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_MENU_ACCOUNT->Name = "BUTTON_MENU_ACCOUNT";
                this->BUTTON_MENU_ACCOUNT->Size = System::Drawing::Size(75, 23);
                this->BUTTON_MENU_ACCOUNT->TabIndex = 2;
                this->BUTTON_MENU_ACCOUNT->Text = "Account";
                this->BUTTON_MENU_ACCOUNT->UseVisualStyleBackColor = true;
                this->BUTTON_MENU_ACCOUNT->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_MENU_ACCOUNTClick);
                // 
                // BUTTON_MENU_TRADE
                // 
                this->BUTTON_MENU_TRADE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_MENU_TRADE->FlatAppearance->BorderSize = 0;
                this->BUTTON_MENU_TRADE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                               static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_MENU_TRADE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                               static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_MENU_TRADE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_MENU_TRADE->ForeColor = System::Drawing::Color::White;
                this->BUTTON_MENU_TRADE->Location = System::Drawing::Point(225, 0);
                this->BUTTON_MENU_TRADE->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_MENU_TRADE->Name = "BUTTON_MENU_TRADE";
                this->BUTTON_MENU_TRADE->Size = System::Drawing::Size(75, 23);
                this->BUTTON_MENU_TRADE->TabIndex = 3;
                this->BUTTON_MENU_TRADE->Text = "Trade";
                this->BUTTON_MENU_TRADE->UseVisualStyleBackColor = true;
                this->BUTTON_MENU_TRADE->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_MENU_TRADE_Click);
                // 
                // OutputTableLayoutPanel
                // 
                this->OutputTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->OutputTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                           static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->OutputTableLayoutPanel->ColumnCount = 1;
                this->OutputTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->OutputTableLayoutPanel->Controls->Add(this->OutputRichTextBox, 0, 1);
                this->OutputTableLayoutPanel->Controls->Add(this->OutputMenuTableLayoutPanel, 0, 0);
                this->OutputTableLayoutPanel->Enabled = false;
                this->OutputTableLayoutPanel->Location = System::Drawing::Point(0, 25);
                this->OutputTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->OutputTableLayoutPanel->Name = "OutputTableLayoutPanel";
                this->OutputTableLayoutPanel->Padding = System::Windows::Forms::Padding(2, 0, 2, 2);
                this->OutputTableLayoutPanel->RowCount = 2;
                this->OutputTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->OutputTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->OutputTableLayoutPanel->Size = System::Drawing::Size(628, 1);
                this->OutputTableLayoutPanel->TabIndex = 9;
                this->OutputTableLayoutPanel->Visible = false;
                // 
                // OutputRichTextBox
                // 
                this->OutputRichTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->OutputRichTextBox->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                      static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->OutputRichTextBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->OutputRichTextBox->Font = (gcnew System::Drawing::Font("Consolas", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->OutputRichTextBox->ForeColor = System::Drawing::Color::White;
                this->OutputRichTextBox->Location = System::Drawing::Point(2, 20);
                this->OutputRichTextBox->Margin = System::Windows::Forms::Padding(0);
                this->OutputRichTextBox->Name = "OutputRichTextBox";
                this->OutputRichTextBox->ReadOnly = true;
                this->OutputRichTextBox->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::ForcedBoth;
                this->OutputRichTextBox->Size = System::Drawing::Size(624, 1);
                this->OutputRichTextBox->TabIndex = 9;
                this->OutputRichTextBox->Text = " ";
                this->OutputRichTextBox->WordWrap = false;
                this->OutputRichTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &CFormMain::OutputRichTextBox_KeyPress);
                // 
                // OutputMenuTableLayoutPanel
                // 
                this->OutputMenuTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->OutputMenuTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                               static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->OutputMenuTableLayoutPanel->ColumnCount = 2;
                this->OutputMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    50)));
                this->OutputMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    50)));
                this->OutputMenuTableLayoutPanel->Location = System::Drawing::Point(2, 0);
                this->OutputMenuTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->OutputMenuTableLayoutPanel->Name = "OutputMenuTableLayoutPanel";
                this->OutputMenuTableLayoutPanel->RowCount = 1;
                this->OutputMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->OutputMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->OutputMenuTableLayoutPanel->Size = System::Drawing::Size(624, 20);
                this->OutputMenuTableLayoutPanel->TabIndex = 10;
                // 
                // SignalTableLayoutPanel
                // 
                this->SignalTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->SignalTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                           static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->SignalTableLayoutPanel->ColumnCount = 1;
                this->SignalTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->SignalTableLayoutPanel->Controls->Add(this->SignalFilterListView, 0, 0);
                this->SignalTableLayoutPanel->Controls->Add(this->SignalListView, 0, 1);
                this->SignalTableLayoutPanel->Location = System::Drawing::Point(0, 25);
                this->SignalTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->SignalTableLayoutPanel->Name = "SignalTableLayoutPanel";
                this->SignalTableLayoutPanel->Padding = System::Windows::Forms::Padding(2, 0, 2, 2);
                this->SignalTableLayoutPanel->RowCount = 2;
                this->SignalTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->SignalTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    40)));
                this->SignalTableLayoutPanel->Size = System::Drawing::Size(628, 1);
                this->SignalTableLayoutPanel->TabIndex = 10;
                // 
                // SignalFilterListView
                // 
                this->SignalFilterListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->SignalFilterListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->SignalFilterListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->SignalFilterListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(8) {
                    this->SignalFilterColumnHeader0,
                        this->SignalFilterColumnHeader1, this->SignalFilterColumnHeader2, this->SignalFilterColumnHeader3, this->SignalFilterColumnHeader4,
                        this->SignalFilterColumnHeader5, this->SignalFilterColumnHeader6, this->SignalFilterColumnHeader7
                });
                this->SignalFilterListView->FullRowSelect = true;
                this->SignalFilterListView->GridLines = true;
                this->SignalFilterListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->SignalFilterListView->LabelWrap = false;
                this->SignalFilterListView->Location = System::Drawing::Point(2, 0);
                this->SignalFilterListView->Margin = System::Windows::Forms::Padding(0);
                this->SignalFilterListView->Name = "SignalFilterListView";
                this->SignalFilterListView->ShowItemToolTips = true;
                this->SignalFilterListView->Size = System::Drawing::Size(624, 1);
                this->SignalFilterListView->TabIndex = 0;
                this->SignalFilterListView->TileSize = System::Drawing::Size(1, 1);
                this->SignalFilterListView->UseCompatibleStateImageBehavior = false;
                this->SignalFilterListView->View = System::Windows::Forms::View::Details;
                // 
                // SignalFilterColumnHeader0
                // 
                this->SignalFilterColumnHeader0->Text = "Signal";
                this->SignalFilterColumnHeader0->Width = 77;
                // 
                // SignalFilterColumnHeader1
                // 
                this->SignalFilterColumnHeader1->Text = "Symbol";
                this->SignalFilterColumnHeader1->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader1->Width = 77;
                // 
                // SignalFilterColumnHeader2
                // 
                this->SignalFilterColumnHeader2->Text = "Time Frames";
                this->SignalFilterColumnHeader2->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader2->Width = 77;
                // 
                // SignalFilterColumnHeader3
                // 
                this->SignalFilterColumnHeader3->Text = "Period Bar(s)";
                this->SignalFilterColumnHeader3->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader3->Width = 77;
                // 
                // SignalFilterColumnHeader4
                // 
                this->SignalFilterColumnHeader4->Text = "Condition(s)";
                this->SignalFilterColumnHeader4->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader4->Width = 77;
                // 
                // SignalFilterColumnHeader5
                // 
                this->SignalFilterColumnHeader5->Text = "Pattern";
                this->SignalFilterColumnHeader5->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader5->Width = 77;
                // 
                // SignalFilterColumnHeader6
                // 
                this->SignalFilterColumnHeader6->Text = "Type Pattern";
                this->SignalFilterColumnHeader6->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader6->Width = 77;
                // 
                // SignalFilterColumnHeader7
                // 
                this->SignalFilterColumnHeader7->Text = "Quality";
                this->SignalFilterColumnHeader7->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalFilterColumnHeader7->Width = 77;
                // 
                // SignalListView
                // 
                this->SignalListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->SignalListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                   static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->SignalListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->SignalListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(6) {
                    this->SignalColumnHeader0,
                        this->SignalColumnHeader1, this->SignalColumnHeader2, this->SignalColumnHeader3, this->SignalColumnHeader4, this->SignalColumnHeader5
                });
                this->SignalListView->FullRowSelect = true;
                this->SignalListView->GridLines = true;
                this->SignalListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->SignalListView->Items->AddRange(gcnew cli::array< System::Windows::Forms::ListViewItem^  >(1) { listViewItem1 });
                this->SignalListView->LabelWrap = false;
                this->SignalListView->Location = System::Drawing::Point(2, -39);
                this->SignalListView->Margin = System::Windows::Forms::Padding(0);
                this->SignalListView->Name = "SignalListView";
                this->SignalListView->Scrollable = false;
                this->SignalListView->ShowItemToolTips = true;
                this->SignalListView->Size = System::Drawing::Size(624, 40);
                this->SignalListView->TabIndex = 1;
                this->SignalListView->UseCompatibleStateImageBehavior = false;
                this->SignalListView->View = System::Windows::Forms::View::Details;
                // 
                // SignalColumnHeader0
                // 
                this->SignalColumnHeader0->Text = "Condition(s)";
                this->SignalColumnHeader0->Width = 103;
                // 
                // SignalColumnHeader1
                // 
                this->SignalColumnHeader1->Text = "Total Signal(s)";
                this->SignalColumnHeader1->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalColumnHeader1->Width = 103;
                // 
                // SignalColumnHeader2
                // 
                this->SignalColumnHeader2->Text = "NB Signal(s) Low";
                this->SignalColumnHeader2->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalColumnHeader2->Width = 103;
                // 
                // SignalColumnHeader3
                // 
                this->SignalColumnHeader3->Text = "NB Signal(s) Med";
                this->SignalColumnHeader3->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalColumnHeader3->Width = 103;
                // 
                // SignalColumnHeader4
                // 
                this->SignalColumnHeader4->Text = "NB Signal(s) High";
                this->SignalColumnHeader4->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalColumnHeader4->Width = 103;
                // 
                // SignalColumnHeader5
                // 
                this->SignalColumnHeader5->Text = "Event Signal Wait";
                this->SignalColumnHeader5->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->SignalColumnHeader5->Width = 103;
                // 
                // AccountTableLayoutPanel
                // 
                this->AccountTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AccountTableLayoutPanel->AutoScroll = true;
                this->AccountTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                            static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->AccountTableLayoutPanel->ColumnCount = 1;
                this->AccountTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->AccountTableLayoutPanel->Controls->Add(this->AccountInfo_Title_Label, 0, 0);
                this->AccountTableLayoutPanel->Controls->Add(this->AccountInfo_Desc_Label, 0, 1);
                this->AccountTableLayoutPanel->Controls->Add(this->PositionInfo_Title_Label, 0, 2);
                this->AccountTableLayoutPanel->Controls->Add(this->PositionInfo_Desc_Label, 0, 3);
                this->AccountTableLayoutPanel->Enabled = false;
                this->AccountTableLayoutPanel->Location = System::Drawing::Point(0, 25);
                this->AccountTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->AccountTableLayoutPanel->Name = "AccountTableLayoutPanel";
                this->AccountTableLayoutPanel->Padding = System::Windows::Forms::Padding(1, 0, 1, 1);
                this->AccountTableLayoutPanel->RowCount = 6;
                this->AccountTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->AccountTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->AccountTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->AccountTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->AccountTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->AccountTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->AccountTableLayoutPanel->Size = System::Drawing::Size(628, 1);
                this->AccountTableLayoutPanel->TabIndex = 13;
                this->AccountTableLayoutPanel->Visible = false;
                // 
                // AccountInfo_Title_Label
                // 
                this->AccountInfo_Title_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AccountInfo_Title_Label->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 11.25F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->AccountInfo_Title_Label->ForeColor = System::Drawing::Color::White;
                this->AccountInfo_Title_Label->Location = System::Drawing::Point(4, 0);
                this->AccountInfo_Title_Label->Name = "AccountInfo_Title_Label";
                this->AccountInfo_Title_Label->Size = System::Drawing::Size(606, 25);
                this->AccountInfo_Title_Label->TabIndex = 0;
                this->AccountInfo_Title_Label->Text = "Account Info";
                this->AccountInfo_Title_Label->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // AccountInfo_Desc_Label
                // 
                this->AccountInfo_Desc_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->AccountInfo_Desc_Label->AutoSize = true;
                this->AccountInfo_Desc_Label->ForeColor = System::Drawing::Color::White;
                this->AccountInfo_Desc_Label->Location = System::Drawing::Point(4, 25);
                this->AccountInfo_Desc_Label->Name = "AccountInfo_Desc_Label";
                this->AccountInfo_Desc_Label->Padding = System::Windows::Forms::Padding(5, 0, 5, 5);
                this->AccountInfo_Desc_Label->Size = System::Drawing::Size(606, 18);
                this->AccountInfo_Desc_Label->TabIndex = 1;
                this->AccountInfo_Desc_Label->Text = "Empty";
                // 
                // PositionInfo_Title_Label
                // 
                this->PositionInfo_Title_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->PositionInfo_Title_Label->Font = (gcnew System::Drawing::Font("Microsoft Sans Serif", 11.25F));
                this->PositionInfo_Title_Label->ForeColor = System::Drawing::Color::White;
                this->PositionInfo_Title_Label->Location = System::Drawing::Point(4, 43);
                this->PositionInfo_Title_Label->Name = "PositionInfo_Title_Label";
                this->PositionInfo_Title_Label->Size = System::Drawing::Size(606, 25);
                this->PositionInfo_Title_Label->TabIndex = 2;
                this->PositionInfo_Title_Label->Text = "Position Info";
                this->PositionInfo_Title_Label->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // PositionInfo_Desc_Label
                // 
                this->PositionInfo_Desc_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->PositionInfo_Desc_Label->AutoSize = true;
                this->PositionInfo_Desc_Label->ForeColor = System::Drawing::Color::White;
                this->PositionInfo_Desc_Label->Location = System::Drawing::Point(4, 68);
                this->PositionInfo_Desc_Label->Name = "PositionInfo_Desc_Label";
                this->PositionInfo_Desc_Label->Padding = System::Windows::Forms::Padding(5, 0, 5, 5);
                this->PositionInfo_Desc_Label->Size = System::Drawing::Size(606, 18);
                this->PositionInfo_Desc_Label->TabIndex = 3;
                this->PositionInfo_Desc_Label->Text = "Empty";
                // 
                // TradeTableLayoutPanel
                // 
                this->TradeTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                                                                                          static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->TradeTableLayoutPanel->ColumnCount = 5;
                this->TradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->TradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->TradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->TradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->TradeTableLayoutPanel->Controls->Add(this->TradeMenuTableLayoutPanel, 0, 0);
                this->TradeTableLayoutPanel->Controls->Add(this->TradeRatioTableLayoutPanel, 4, 0);
                this->TradeTableLayoutPanel->Controls->Add(this->TradeHistoryTableLayoutPanel, 2, 0);
                this->TradeTableLayoutPanel->Controls->Add(this->TradeTradeTableLayoutPanel, 1, 0);
                this->TradeTableLayoutPanel->Controls->Add(this->TradeOperationsTableLayoutPanel, 3, 0);
                this->TradeTableLayoutPanel->Enabled = false;
                this->TradeTableLayoutPanel->Location = System::Drawing::Point(0, 25);
                this->TradeTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeTableLayoutPanel->Name = "TradeTableLayoutPanel";
                this->TradeTableLayoutPanel->Padding = System::Windows::Forms::Padding(2, 0, 2, 2);
                this->TradeTableLayoutPanel->RowCount = 1;
                this->TradeTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeTableLayoutPanel->Size = System::Drawing::Size(628, 360);
                this->TradeTableLayoutPanel->TabIndex = 12;
                this->TradeTableLayoutPanel->Visible = false;
                // 
                // TradeMenuTableLayoutPanel
                // 
                this->TradeMenuTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeMenuTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                              static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->TradeMenuTableLayoutPanel->ColumnCount = 1;
                this->TradeMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeMenuTableLayoutPanel->Controls->Add(this->BUTTON_TRADE_HISTORY, 0, 1);
                this->TradeMenuTableLayoutPanel->Controls->Add(this->BUTTON_TRADE_TRADE, 0, 0);
                this->TradeMenuTableLayoutPanel->Controls->Add(this->BUTTON_TRADE_RATIO, 0, 3);
                this->TradeMenuTableLayoutPanel->Controls->Add(this->BUTTON_TRADE_OPERATIONS, 0, 2);
                this->TradeMenuTableLayoutPanel->Location = System::Drawing::Point(2, 0);
                this->TradeMenuTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeMenuTableLayoutPanel->Name = "TradeMenuTableLayoutPanel";
                this->TradeMenuTableLayoutPanel->RowCount = 4;
                this->TradeMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    25)));
                this->TradeMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    25)));
                this->TradeMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    25)));
                this->TradeMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    25)));
                this->TradeMenuTableLayoutPanel->Size = System::Drawing::Size(25, 358);
                this->TradeMenuTableLayoutPanel->TabIndex = 2;
                // 
                // BUTTON_TRADE_HISTORY
                // 
                this->BUTTON_TRADE_HISTORY->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_TRADE_HISTORY->FlatAppearance->BorderSize = 0;
                this->BUTTON_TRADE_HISTORY->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                  static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_TRADE_HISTORY->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                  static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_TRADE_HISTORY->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_TRADE_HISTORY->ForeColor = System::Drawing::Color::White;
                this->BUTTON_TRADE_HISTORY->Location = System::Drawing::Point(0, 89);
                this->BUTTON_TRADE_HISTORY->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_TRADE_HISTORY->Name = "BUTTON_TRADE_HISTORY";
                this->BUTTON_TRADE_HISTORY->Size = System::Drawing::Size(25, 89);
                this->BUTTON_TRADE_HISTORY->TabIndex = 0;
                this->BUTTON_TRADE_HISTORY->Text = "H\r\nI\r\nS\r\nT\r\nO\r\nR\r\nY";
                this->BUTTON_TRADE_HISTORY->UseVisualStyleBackColor = true;
                this->BUTTON_TRADE_HISTORY->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_TRADE_HISTORY_Click);
                // 
                // BUTTON_TRADE_TRADE
                // 
                this->BUTTON_TRADE_TRADE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_TRADE_TRADE->FlatAppearance->BorderSize = 0;
                this->BUTTON_TRADE_TRADE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_TRADE_TRADE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_TRADE_TRADE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_TRADE_TRADE->ForeColor = System::Drawing::Color::White;
                this->BUTTON_TRADE_TRADE->Location = System::Drawing::Point(0, 0);
                this->BUTTON_TRADE_TRADE->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_TRADE_TRADE->Name = "BUTTON_TRADE_TRADE";
                this->BUTTON_TRADE_TRADE->Size = System::Drawing::Size(25, 89);
                this->BUTTON_TRADE_TRADE->TabIndex = 2;
                this->BUTTON_TRADE_TRADE->Text = "T\r\nR\r\nA\r\nD\r\nE";
                this->BUTTON_TRADE_TRADE->UseVisualStyleBackColor = true;
                this->BUTTON_TRADE_TRADE->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_TRADE_TRADE_Click);
                // 
                // BUTTON_TRADE_RATIO
                // 
                this->BUTTON_TRADE_RATIO->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_TRADE_RATIO->FlatAppearance->BorderSize = 0;
                this->BUTTON_TRADE_RATIO->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_TRADE_RATIO->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_TRADE_RATIO->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_TRADE_RATIO->ForeColor = System::Drawing::Color::White;
                this->BUTTON_TRADE_RATIO->Location = System::Drawing::Point(0, 267);
                this->BUTTON_TRADE_RATIO->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_TRADE_RATIO->Name = "BUTTON_TRADE_RATIO";
                this->BUTTON_TRADE_RATIO->Size = System::Drawing::Size(25, 91);
                this->BUTTON_TRADE_RATIO->TabIndex = 3;
                this->BUTTON_TRADE_RATIO->Text = "R\r\nA\r\nT\r\nI\r\nO";
                this->BUTTON_TRADE_RATIO->UseVisualStyleBackColor = true;
                this->BUTTON_TRADE_RATIO->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_TRADE_RATIO_Click);
                // 
                // BUTTON_TRADE_OPERATIONS
                // 
                this->BUTTON_TRADE_OPERATIONS->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_TRADE_OPERATIONS->FlatAppearance->BorderSize = 0;
                this->BUTTON_TRADE_OPERATIONS->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                     static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_TRADE_OPERATIONS->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                     static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_TRADE_OPERATIONS->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_TRADE_OPERATIONS->ForeColor = System::Drawing::Color::White;
                this->BUTTON_TRADE_OPERATIONS->Location = System::Drawing::Point(0, 178);
                this->BUTTON_TRADE_OPERATIONS->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_TRADE_OPERATIONS->Name = "BUTTON_TRADE_OPERATIONS";
                this->BUTTON_TRADE_OPERATIONS->Size = System::Drawing::Size(25, 89);
                this->BUTTON_TRADE_OPERATIONS->TabIndex = 1;
                this->BUTTON_TRADE_OPERATIONS->Text = "O\r\nP\r\nE\r\nR\r\nA\r\nT\r\nI\r\nO\r\nN\r\nS";
                this->BUTTON_TRADE_OPERATIONS->UseVisualStyleBackColor = true;
                this->BUTTON_TRADE_OPERATIONS->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_TRADE_OPERATIONS_Click);
                // 
                // TradeRatioTableLayoutPanel
                // 
                this->TradeRatioTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeRatioTableLayoutPanel->ColumnCount = 1;
                this->TradeRatioTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeRatioTableLayoutPanel->Controls->Add(this->TradeTypeRatioTableLayoutPanel, 0, 1);
                this->TradeRatioTableLayoutPanel->Controls->Add(this->TradeRatioMenuTableLayoutPanel, 0, 0);
                this->TradeRatioTableLayoutPanel->Enabled = false;
                this->TradeRatioTableLayoutPanel->Location = System::Drawing::Point(626, 0);
                this->TradeRatioTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeRatioTableLayoutPanel->Name = "TradeRatioTableLayoutPanel";
                this->TradeRatioTableLayoutPanel->RowCount = 2;
                this->TradeRatioTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->TradeRatioTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeRatioTableLayoutPanel->Size = System::Drawing::Size(1, 358);
                this->TradeRatioTableLayoutPanel->TabIndex = 4;
                this->TradeRatioTableLayoutPanel->Visible = false;
                // 
                // TradeTypeRatioTableLayoutPanel
                // 
                this->TradeTypeRatioTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeTypeRatioTableLayoutPanel->ColumnCount = 1;
                this->TradeTypeRatioTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeTypeRatioTableLayoutPanel->Controls->Add(this->TradeRatioFloorListView, 0, 0);
                this->TradeTypeRatioTableLayoutPanel->Controls->Add(this->TradeRatioRoundListView, 0, 1);
                this->TradeTypeRatioTableLayoutPanel->Location = System::Drawing::Point(0, 25);
                this->TradeTypeRatioTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeTypeRatioTableLayoutPanel->Name = "TradeTypeRatioTableLayoutPanel";
                this->TradeTypeRatioTableLayoutPanel->RowCount = 2;
                this->TradeTypeRatioTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeTypeRatioTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->TradeTypeRatioTableLayoutPanel->Size = System::Drawing::Size(1, 333);
                this->TradeTypeRatioTableLayoutPanel->TabIndex = 0;
                // 
                // TradeRatioFloorListView
                // 
                this->TradeRatioFloorListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeRatioFloorListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                            static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->TradeRatioFloorListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->TradeRatioFloorListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(5) {
                    this->columnHeader48,
                        this->columnHeader49, this->columnHeader50, this->columnHeader51, this->columnHeader52
                });
                this->TradeRatioFloorListView->ForeColor = System::Drawing::Color::White;
                this->TradeRatioFloorListView->FullRowSelect = true;
                this->TradeRatioFloorListView->GridLines = true;
                this->TradeRatioFloorListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->TradeRatioFloorListView->Location = System::Drawing::Point(0, 0);
                this->TradeRatioFloorListView->Margin = System::Windows::Forms::Padding(0);
                this->TradeRatioFloorListView->Name = "TradeRatioFloorListView";
                this->TradeRatioFloorListView->Size = System::Drawing::Size(1, 333);
                this->TradeRatioFloorListView->TabIndex = 8;
                this->TradeRatioFloorListView->UseCompatibleStateImageBehavior = false;
                this->TradeRatioFloorListView->View = System::Windows::Forms::View::Details;
                // 
                // columnHeader48
                // 
                this->columnHeader48->Text = "Symbol";
                this->columnHeader48->Width = 119;
                // 
                // columnHeader49
                // 
                this->columnHeader49->Text = "Condition(s)";
                this->columnHeader49->Width = 119;
                // 
                // columnHeader50
                // 
                this->columnHeader50->Text = "Count Win";
                this->columnHeader50->Width = 119;
                // 
                // columnHeader51
                // 
                this->columnHeader51->Text = "Count Loss";
                this->columnHeader51->Width = 119;
                // 
                // columnHeader52
                // 
                this->columnHeader52->Text = "Count Drawn";
                this->columnHeader52->Width = 119;
                // 
                // TradeRatioRoundListView
                // 
                this->TradeRatioRoundListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeRatioRoundListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                            static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->TradeRatioRoundListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->TradeRatioRoundListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(5) {
                    this->columnHeader38,
                        this->columnHeader39, this->columnHeader40, this->columnHeader41, this->columnHeader42
                });
                this->TradeRatioRoundListView->ForeColor = System::Drawing::Color::White;
                this->TradeRatioRoundListView->FullRowSelect = true;
                this->TradeRatioRoundListView->GridLines = true;
                this->TradeRatioRoundListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->TradeRatioRoundListView->Location = System::Drawing::Point(0, 333);
                this->TradeRatioRoundListView->Margin = System::Windows::Forms::Padding(0);
                this->TradeRatioRoundListView->Name = "TradeRatioRoundListView";
                this->TradeRatioRoundListView->Size = System::Drawing::Size(1, 1);
                this->TradeRatioRoundListView->TabIndex = 6;
                this->TradeRatioRoundListView->UseCompatibleStateImageBehavior = false;
                this->TradeRatioRoundListView->View = System::Windows::Forms::View::Details;
                // 
                // columnHeader38
                // 
                this->columnHeader38->Text = "Symbol";
                this->columnHeader38->Width = 119;
                // 
                // columnHeader39
                // 
                this->columnHeader39->Text = "Condition(s)";
                this->columnHeader39->Width = 119;
                // 
                // columnHeader40
                // 
                this->columnHeader40->Text = "Count Win";
                this->columnHeader40->Width = 119;
                // 
                // columnHeader41
                // 
                this->columnHeader41->Text = "Count Loss";
                this->columnHeader41->Width = 119;
                // 
                // columnHeader42
                // 
                this->columnHeader42->Text = "Count Drawn";
                this->columnHeader42->Width = 119;
                // 
                // TradeRatioMenuTableLayoutPanel
                // 
                this->TradeRatioMenuTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeRatioMenuTableLayoutPanel->ColumnCount = 3;
                this->TradeRatioMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    125)));
                this->TradeRatioMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    125)));
                this->TradeRatioMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeRatioMenuTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->TradeRatioMenuTableLayoutPanel->Controls->Add(this->BUTTON_RATIO_FLOOR, 0, 0);
                this->TradeRatioMenuTableLayoutPanel->Controls->Add(this->BUTTON_RATIO_ROUND, 1, 0);
                this->TradeRatioMenuTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->TradeRatioMenuTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeRatioMenuTableLayoutPanel->Name = "TradeRatioMenuTableLayoutPanel";
                this->TradeRatioMenuTableLayoutPanel->RowCount = 1;
                this->TradeRatioMenuTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeRatioMenuTableLayoutPanel->Size = System::Drawing::Size(1, 25);
                this->TradeRatioMenuTableLayoutPanel->TabIndex = 1;
                // 
                // BUTTON_RATIO_FLOOR
                // 
                this->BUTTON_RATIO_FLOOR->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_RATIO_FLOOR->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                       static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->BUTTON_RATIO_FLOOR->FlatAppearance->BorderSize = 0;
                this->BUTTON_RATIO_FLOOR->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_RATIO_FLOOR->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_RATIO_FLOOR->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_RATIO_FLOOR->ForeColor = System::Drawing::Color::White;
                this->BUTTON_RATIO_FLOOR->Location = System::Drawing::Point(0, 0);
                this->BUTTON_RATIO_FLOOR->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_RATIO_FLOOR->Name = "BUTTON_RATIO_FLOOR";
                this->BUTTON_RATIO_FLOOR->Size = System::Drawing::Size(125, 25);
                this->BUTTON_RATIO_FLOOR->TabIndex = 0;
                this->BUTTON_RATIO_FLOOR->Text = "Floor";
                this->BUTTON_RATIO_FLOOR->UseVisualStyleBackColor = false;
                this->BUTTON_RATIO_FLOOR->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_RATIO_FLOOR_CLICK);
                // 
                // BUTTON_RATIO_ROUND
                // 
                this->BUTTON_RATIO_ROUND->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BUTTON_RATIO_ROUND->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                       static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->BUTTON_RATIO_ROUND->FlatAppearance->BorderSize = 0;
                this->BUTTON_RATIO_ROUND->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BUTTON_RATIO_ROUND->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(28)),
                                                                                                                static_cast<System::Int32>(static_cast<System::Byte>(151)), static_cast<System::Int32>(static_cast<System::Byte>(234)));
                this->BUTTON_RATIO_ROUND->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->BUTTON_RATIO_ROUND->ForeColor = System::Drawing::Color::White;
                this->BUTTON_RATIO_ROUND->Location = System::Drawing::Point(125, 0);
                this->BUTTON_RATIO_ROUND->Margin = System::Windows::Forms::Padding(0);
                this->BUTTON_RATIO_ROUND->Name = "BUTTON_RATIO_ROUND";
                this->BUTTON_RATIO_ROUND->Size = System::Drawing::Size(125, 25);
                this->BUTTON_RATIO_ROUND->TabIndex = 1;
                this->BUTTON_RATIO_ROUND->Text = "Round";
                this->BUTTON_RATIO_ROUND->UseVisualStyleBackColor = false;
                this->BUTTON_RATIO_ROUND->Click += gcnew System::EventHandler(this, &CFormMain::BUTTON_RATIO_ROUND_CLICK);
                // 
                // TradeHistoryTableLayoutPanel
                // 
                this->TradeHistoryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeHistoryTableLayoutPanel->ColumnCount = 1;
                this->TradeHistoryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeHistoryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->TradeHistoryTableLayoutPanel->Controls->Add(this->TradeHistoryListView, 0, 0);
                this->TradeHistoryTableLayoutPanel->Enabled = false;
                this->TradeHistoryTableLayoutPanel->Location = System::Drawing::Point(626, 0);
                this->TradeHistoryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeHistoryTableLayoutPanel->Name = "TradeHistoryTableLayoutPanel";
                this->TradeHistoryTableLayoutPanel->RowCount = 1;
                this->TradeHistoryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeHistoryTableLayoutPanel->Size = System::Drawing::Size(1, 358);
                this->TradeHistoryTableLayoutPanel->TabIndex = 5;
                this->TradeHistoryTableLayoutPanel->Visible = false;
                // 
                // TradeHistoryListView
                // 
                this->TradeHistoryListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeHistoryListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                         static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->TradeHistoryListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->TradeHistoryListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(11) {
                    this->columnHeader1,
                        this->columnHeader2, this->columnHeader3, this->columnHeader4, this->columnHeader5, this->columnHeader6, this->columnHeader7,
                        this->columnHeader8, this->columnHeader9, this->columnHeader10, this->columnHeader11
                });
                this->TradeHistoryListView->FullRowSelect = true;
                this->TradeHistoryListView->GridLines = true;
                this->TradeHistoryListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->TradeHistoryListView->LabelWrap = false;
                this->TradeHistoryListView->Location = System::Drawing::Point(0, 0);
                this->TradeHistoryListView->Margin = System::Windows::Forms::Padding(0);
                this->TradeHistoryListView->Name = "TradeHistoryListView";
                this->TradeHistoryListView->ShowItemToolTips = true;
                this->TradeHistoryListView->Size = System::Drawing::Size(1, 358);
                this->TradeHistoryListView->TabIndex = 0;
                this->TradeHistoryListView->UseCompatibleStateImageBehavior = false;
                this->TradeHistoryListView->View = System::Windows::Forms::View::Details;
                // 
                // columnHeader1
                // 
                this->columnHeader1->Text = "Time";
                this->columnHeader1->Width = 57;
                // 
                // columnHeader2
                // 
                this->columnHeader2->Text = "Order";
                this->columnHeader2->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader2->Width = 57;
                // 
                // columnHeader3
                // 
                this->columnHeader3->Text = "Symbol";
                this->columnHeader3->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader3->Width = 57;
                // 
                // columnHeader4
                // 
                this->columnHeader4->Text = "Type";
                this->columnHeader4->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader4->Width = 57;
                // 
                // columnHeader5
                // 
                this->columnHeader5->Text = "Volume";
                this->columnHeader5->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader5->Width = 57;
                // 
                // columnHeader6
                // 
                this->columnHeader6->Text = "Price";
                this->columnHeader6->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader6->Width = 57;
                // 
                // columnHeader7
                // 
                this->columnHeader7->Text = "S / ";
                this->columnHeader7->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader7->Width = 57;
                // 
                // columnHeader8
                // 
                this->columnHeader8->Text = "T / P";
                this->columnHeader8->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader8->Width = 57;
                // 
                // columnHeader9
                // 
                this->columnHeader9->Text = "Time";
                this->columnHeader9->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader9->Width = 57;
                // 
                // columnHeader10
                // 
                this->columnHeader10->Text = "State";
                this->columnHeader10->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader10->Width = 57;
                // 
                // columnHeader11
                // 
                this->columnHeader11->Text = "Comment";
                this->columnHeader11->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader11->Width = 57;
                // 
                // TradeTradeTableLayoutPanel
                // 
                this->TradeTradeTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeTradeTableLayoutPanel->ColumnCount = 1;
                this->TradeTradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeTradeTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->TradeTradeTableLayoutPanel->Controls->Add(this->TradeTradeListView, 0, 0);
                this->TradeTradeTableLayoutPanel->Location = System::Drawing::Point(27, 0);
                this->TradeTradeTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeTradeTableLayoutPanel->Name = "TradeTradeTableLayoutPanel";
                this->TradeTradeTableLayoutPanel->RowCount = 1;
                this->TradeTradeTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeTradeTableLayoutPanel->Size = System::Drawing::Size(599, 358);
                this->TradeTradeTableLayoutPanel->TabIndex = 6;
                // 
                // TradeTradeListView
                // 
                this->TradeTradeListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeTradeListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                       static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->TradeTradeListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->TradeTradeListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(11) {
                    this->columnHeader22,
                        this->columnHeader23, this->columnHeader24, this->columnHeader25, this->columnHeader26, this->columnHeader27, this->columnHeader28,
                        this->columnHeader29, this->columnHeader30, this->columnHeader31, this->columnHeader32
                });
                this->TradeTradeListView->FullRowSelect = true;
                this->TradeTradeListView->GridLines = true;
                this->TradeTradeListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->TradeTradeListView->LabelWrap = false;
                this->TradeTradeListView->Location = System::Drawing::Point(0, 0);
                this->TradeTradeListView->Margin = System::Windows::Forms::Padding(0);
                this->TradeTradeListView->Name = "TradeTradeListView";
                this->TradeTradeListView->Size = System::Drawing::Size(599, 358);
                this->TradeTradeListView->TabIndex = 3;
                this->TradeTradeListView->UseCompatibleStateImageBehavior = false;
                this->TradeTradeListView->View = System::Windows::Forms::View::Details;
                // 
                // columnHeader22
                // 
                this->columnHeader22->Text = "Symbol";
                this->columnHeader22->Width = 53;
                // 
                // columnHeader23
                // 
                this->columnHeader23->Text = "Ticket";
                this->columnHeader23->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader23->Width = 53;
                // 
                // columnHeader24
                // 
                this->columnHeader24->Text = "Time";
                this->columnHeader24->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader24->Width = 53;
                // 
                // columnHeader25
                // 
                this->columnHeader25->Text = "Type";
                this->columnHeader25->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader25->Width = 53;
                // 
                // columnHeader26
                // 
                this->columnHeader26->Text = "Volume";
                this->columnHeader26->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader26->Width = 53;
                // 
                // columnHeader27
                // 
                this->columnHeader27->Text = "Price";
                this->columnHeader27->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader27->Width = 53;
                // 
                // columnHeader28
                // 
                this->columnHeader28->Text = "S / ";
                this->columnHeader28->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader28->Width = 53;
                // 
                // columnHeader29
                // 
                this->columnHeader29->Text = "T / P";
                this->columnHeader29->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader29->Width = 53;
                // 
                // columnHeader30
                // 
                this->columnHeader30->Text = "Price";
                this->columnHeader30->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader30->Width = 53;
                // 
                // columnHeader31
                // 
                this->columnHeader31->Text = "Profit";
                this->columnHeader31->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader31->Width = 53;
                // 
                // columnHeader32
                // 
                this->columnHeader32->Text = "Comment";
                this->columnHeader32->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                // 
                // TradeOperationsTableLayoutPanel
                // 
                this->TradeOperationsTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeOperationsTableLayoutPanel->ColumnCount = 1;
                this->TradeOperationsTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeOperationsTableLayoutPanel->Controls->Add(this->TradeOperationsListView, 0, 0);
                this->TradeOperationsTableLayoutPanel->Enabled = false;
                this->TradeOperationsTableLayoutPanel->Location = System::Drawing::Point(626, 0);
                this->TradeOperationsTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->TradeOperationsTableLayoutPanel->Name = "TradeOperationsTableLayoutPanel";
                this->TradeOperationsTableLayoutPanel->RowCount = 1;
                this->TradeOperationsTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->TradeOperationsTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    358)));
                this->TradeOperationsTableLayoutPanel->Size = System::Drawing::Size(1, 358);
                this->TradeOperationsTableLayoutPanel->TabIndex = 7;
                this->TradeOperationsTableLayoutPanel->Visible = false;
                // 
                // TradeOperationsListView
                // 
                this->TradeOperationsListView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TradeOperationsListView->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(37)),
                                                                                            static_cast<System::Int32>(static_cast<System::Byte>(37)), static_cast<System::Int32>(static_cast<System::Byte>(38)));
                this->TradeOperationsListView->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->TradeOperationsListView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(10) {
                    this->columnHeader12,
                        this->columnHeader13, this->columnHeader14, this->columnHeader15, this->columnHeader16, this->columnHeader17, this->columnHeader18,
                        this->columnHeader19, this->columnHeader20, this->columnHeader21
                });
                this->TradeOperationsListView->FullRowSelect = true;
                this->TradeOperationsListView->GridLines = true;
                this->TradeOperationsListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
                this->TradeOperationsListView->LabelWrap = false;
                this->TradeOperationsListView->Location = System::Drawing::Point(0, 0);
                this->TradeOperationsListView->Margin = System::Windows::Forms::Padding(0);
                this->TradeOperationsListView->Name = "TradeOperationsListView";
                this->TradeOperationsListView->Size = System::Drawing::Size(1, 358);
                this->TradeOperationsListView->TabIndex = 1;
                this->TradeOperationsListView->UseCompatibleStateImageBehavior = false;
                this->TradeOperationsListView->View = System::Windows::Forms::View::Details;
                // 
                // columnHeader12
                // 
                this->columnHeader12->Text = "Time";
                this->columnHeader12->Width = 55;
                // 
                // columnHeader13
                // 
                this->columnHeader13->Text = "Ticket";
                this->columnHeader13->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader13->Width = 55;
                // 
                // columnHeader14
                // 
                this->columnHeader14->Text = "Symbol";
                this->columnHeader14->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader14->Width = 55;
                // 
                // columnHeader15
                // 
                this->columnHeader15->Text = "Action";
                this->columnHeader15->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader15->Width = 55;
                // 
                // columnHeader16
                // 
                this->columnHeader16->Text = "Type";
                this->columnHeader16->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader16->Width = 55;
                // 
                // columnHeader17
                // 
                this->columnHeader17->Text = "Volume";
                this->columnHeader17->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader17->Width = 55;
                // 
                // columnHeader18
                // 
                this->columnHeader18->Text = "Price";
                this->columnHeader18->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader18->Width = 55;
                // 
                // columnHeader19
                // 
                this->columnHeader19->Text = "S / ";
                this->columnHeader19->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader19->Width = 55;
                // 
                // columnHeader20
                // 
                this->columnHeader20->Text = "T / P";
                this->columnHeader20->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader20->Width = 55;
                // 
                // columnHeader21
                // 
                this->columnHeader21->Text = "Comment";
                this->columnHeader21->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
                this->columnHeader21->Width = 55;
                // 
                // TopBorderPanel
                // 
                this->TopBorderPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->TopBorderPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)), static_cast<System::Int32>(static_cast<System::Byte>(122)),
                                                                                   static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->TopBorderPanel->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->TopBorderPanel->Cursor = System::Windows::Forms::Cursors::SizeNS;
                this->TopBorderPanel->Location = System::Drawing::Point(1, 0);
                this->TopBorderPanel->Margin = System::Windows::Forms::Padding(0);
                this->TopBorderPanel->Name = "TopBorderPanel";
                this->TopBorderPanel->Size = System::Drawing::Size(638, 1);
                this->TopBorderPanel->TabIndex = 0;
                this->TopBorderPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseDown);
                this->TopBorderPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseMove_Top);
                this->TopBorderPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseUp);
                // 
                // FormMainPrimaryTableLayoutPanel
                // 
                this->FormMainPrimaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->FormMainPrimaryTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                                    static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->FormMainPrimaryTableLayoutPanel->ColumnCount = 3;
                this->FormMainPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->FormMainPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormMainPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->FormMainPrimaryTableLayoutPanel->Controls->Add(this->TopBorderPanel, 1, 0);
                this->FormMainPrimaryTableLayoutPanel->Controls->Add(this->RightBorderPanel, 2, 1);
                this->FormMainPrimaryTableLayoutPanel->Controls->Add(this->LeftBorderPanel, 0, 1);
                this->FormMainPrimaryTableLayoutPanel->Controls->Add(this->FormMainSecondaryTableLayoutPanel, 1, 1);
                this->FormMainPrimaryTableLayoutPanel->Controls->Add(this->BottomBorderPanel, 1, 2);
                this->FormMainPrimaryTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->FormMainPrimaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->FormMainPrimaryTableLayoutPanel->Name = "FormMainPrimaryTableLayoutPanel";
                this->FormMainPrimaryTableLayoutPanel->RowCount = 3;
                this->FormMainPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->FormMainPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormMainPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->FormMainPrimaryTableLayoutPanel->Size = System::Drawing::Size(640, 480);
                this->FormMainPrimaryTableLayoutPanel->TabIndex = 6;
                // 
                // FormMainSecondaryTableLayoutPanel
                // 
                this->FormMainSecondaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->FormMainSecondaryTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                                                      static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->FormMainSecondaryTableLayoutPanel->ColumnCount = 1;
                this->FormMainSecondaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormMainSecondaryTableLayoutPanel->Controls->Add(this->StatusTableLayoutPanel, 0, 3);
                this->FormMainSecondaryTableLayoutPanel->Controls->Add(this->FormMenuTableLayoutPanel, 0, 1);
                this->FormMainSecondaryTableLayoutPanel->Controls->Add(this->FormTopTableLayoutPanel, 0, 0);
                this->FormMainSecondaryTableLayoutPanel->Controls->Add(this->MainTableLayoutPanel, 0, 2);
                this->FormMainSecondaryTableLayoutPanel->Location = System::Drawing::Point(1, 1);
                this->FormMainSecondaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->FormMainSecondaryTableLayoutPanel->Name = "FormMainSecondaryTableLayoutPanel";
                this->FormMainSecondaryTableLayoutPanel->RowCount = 4;
                this->FormMainSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->FormMainSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    28)));
                this->FormMainSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->FormMainSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    25)));
                this->FormMainSecondaryTableLayoutPanel->Size = System::Drawing::Size(638, 478);
                this->FormMainSecondaryTableLayoutPanel->TabIndex = 1;
                // 
                // BottomBorderPanel
                // 
                this->BottomBorderPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->BottomBorderPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)), static_cast<System::Int32>(static_cast<System::Byte>(122)),
                                                                                      static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->BottomBorderPanel->Cursor = System::Windows::Forms::Cursors::SizeNS;
                this->BottomBorderPanel->Location = System::Drawing::Point(1, 479);
                this->BottomBorderPanel->Margin = System::Windows::Forms::Padding(0);
                this->BottomBorderPanel->Name = "BottomBorderPanel";
                this->BottomBorderPanel->Size = System::Drawing::Size(638, 1);
                this->BottomBorderPanel->TabIndex = 0;
                this->BottomBorderPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseDown);
                this->BottomBorderPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseMove_Bottom);
                this->BottomBorderPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormMain::FormMain_ReSize_MouseUp);
                // 
                // CFormMain
                // 
                this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
                this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
                this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
                                                                   static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->ClientSize = System::Drawing::Size(640, 480);
                this->Controls->Add(this->FormMainPrimaryTableLayoutPanel);
                this->DoubleBuffered = true;
                this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::None;
                this->MinimumSize = System::Drawing::Size(640, 480);
                this->Name = "CFormMain";
                this->RightToLeft = System::Windows::Forms::RightToLeft::No;
                this->RightToLeftLayout = true;
                this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Show;
                this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
                this->Text = "MyEA";
                this->Load += gcnew System::EventHandler(this, &CFormMain::FormMain_Load);
                this->FormTopTableLayoutPanel->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->LOGO))->EndInit();
                this->FormMenuTableLayoutPanel->ResumeLayout(false);
                this->Help_MenuStrip->ResumeLayout(false);
                this->Help_MenuStrip->PerformLayout();
                this->StatusTableLayoutPanel->ResumeLayout(false);
                this->MainTableLayoutPanel->ResumeLayout(false);
                this->MainMenuTableLayoutPanel->ResumeLayout(false);
                this->MainMenuButtonTableLayoutPanel->ResumeLayout(false);
                this->OutputTableLayoutPanel->ResumeLayout(false);
                this->SignalTableLayoutPanel->ResumeLayout(false);
                this->AccountTableLayoutPanel->ResumeLayout(false);
                this->AccountTableLayoutPanel->PerformLayout();
                this->TradeTableLayoutPanel->ResumeLayout(false);
                this->TradeMenuTableLayoutPanel->ResumeLayout(false);
                this->TradeRatioTableLayoutPanel->ResumeLayout(false);
                this->TradeTypeRatioTableLayoutPanel->ResumeLayout(false);
                this->TradeRatioMenuTableLayoutPanel->ResumeLayout(false);
                this->TradeHistoryTableLayoutPanel->ResumeLayout(false);
                this->TradeTradeTableLayoutPanel->ResumeLayout(false);
                this->TradeOperationsTableLayoutPanel->ResumeLayout(false);
                this->FormMainPrimaryTableLayoutPanel->ResumeLayout(false);
                this->FormMainSecondaryTableLayoutPanel->ResumeLayout(false);
                this->ResumeLayout(false);

            }
#pragma endregion

            // CFormMain
            private: System::Void FormMain_Load(System::Object^  sender, System::EventArgs^  e);
            private: System::Void FormMain_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
            private: System::Void FormMain_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
            private: System::Void FormMain_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
            private: System::Void FormMain_ReSize_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);

            private: System::Void FormMain_ReSize_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);

            // BUTTON_CLOSE
            private: System::Void BUTTON_CLOSE_Click(System::Object^  sender, System::EventArgs^  e);

            // BUTTON_MAXIMIZE
            private: System::Void BUTTON_MAXIMIZE_Click(System::Object^  sender, System::EventArgs^  e);

            // BUTTON_MINIMIZE
            private: System::Void BUTTON_MINIMIZE_Click(System::Object^  sender, System::EventArgs^  e);

            // BUTTON_EXIT
            private: System::Void BUTTON_FILE_Click(System::Object^  sender, System::EventArgs^  e);

            // Maximized_DoubleClick
            private: System::Void Maximized_DoubleClick(System::Object^  sender, System::EventArgs^  e);
    
            // OutputRichTextBox
            private: System::Void OutputRichTextBox_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);

            // BUTTON_VIEW
            private: System::Void BUTTON_VIEW_Click(System::Object^  sender, System::EventArgs^  e);
            
            // BUTTON_MENU_OUTPUT
            private: System::Void BUTTON_MENU_OUTPUTClick(System::Object^  sender, System::EventArgs^  e);
            
            // BUTTON_MENU_SIGNAL
            private: System::Void BUTTON_MENU_SIGNAL_Click(System::Object^  sender, System::EventArgs^  e);
    
            // BUTTON_MENU_ACCOUNT
            private: System::Void BUTTON_MENU_ACCOUNTClick(System::Object^  sender, System::EventArgs^  e);
            
            // BUTTON_MENU_TRADE
            private: System::Void BUTTON_MENU_TRADE_Click(System::Object^  sender, System::EventArgs^  e);

            private: System::Void FormMain_ReSize_MouseMove_Top(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
            private: System::Void FormMain_ReSize_MouseMove_Right(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
            private: System::Void FormMain_ReSize_MouseMove_Left(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
            private: System::Void FormMain_ReSize_MouseMove_Bottom(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);

            private: System::Void AboutMyEAToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
            private: System::Void BUTTON_RATIO_FLOOR_CLICK(System::Object^  sender, System::EventArgs^  e);
            private: System::Void BUTTON_RATIO_ROUND_CLICK(System::Object^  sender, System::EventArgs^  e);
            private: System::Void BUTTON_TRADE_RATIO_Click(System::Object^  sender, System::EventArgs^  e);
            private: System::Void BUTTON_TRADE_HISTORY_Click(System::Object^  sender, System::EventArgs^  e);
            private: System::Void BUTTON_TRADE_TRADE_Click(System::Object^  sender, System::EventArgs^  e);
            private: System::Void BUTTON_TRADE_OPERATIONS_Click(System::Object^  sender, System::EventArgs^  e);
};
    }
}
