#include "stdafx.hpp"

#include <Form/CFormMain.h>
#include <Form/CFormLogin.h>

#include <Tools/Time.hpp>

#include <Files/File.hpp>

namespace MyEA
{
    namespace Form
    {
        static void GetProjectAddr(void) {}

        /* MAIN_CFM */
        const int MAIN_CFM(CFormMain^ ptr_CFM_received)
        {
            // Enabling Windows XP visual effects before any controls are created
            Application::EnableVisualStyles();
            Application::SetCompatibleTextRenderingDefault(false);

            Application::Run(ptr_CFM_received);

            return(0);
        }

        /* CFormMain */
        CFormMain::CFormMain(void) : isThreading(true),
                                                            _moving(false),
                                                            _sizing(false),
                                                            p_min_form_width(640),
                                                            p_min_form_height(480),
                                                            _position_offset(0, 0),
                                                            p_type_main_panel(Common::TYPE_M_P_NONE),
                                                            p_Ptr_l_indicator(gcnew List<int>()),
                                                            p_Ptr_Ressource_Manager(gcnew ResourceManager("FormWin.Resource_Files.Resource", GetType()->Assembly)),
                                                            p_Ptr_CFA(nullptr)
        {
            InitializeComponent();

            //
            //TODO: Add the constructor code here
            //

            // Icon
            this->Icon = safe_cast<System::Drawing::Icon^>(p_Ptr_Ressource_Manager->GetObject("COMPANY_favicon_64_ICO"));
            this->Icon = safe_cast<System::Drawing::Icon^>(p_Ptr_Ressource_Manager->GetObject("COMPANY_favicon_64_ICO"));
        
            // Image
            this->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Background_64_PNG"));
            this->TopBorderPanel->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Border_30_PNG"));
            this->LOGO->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("COMPANY_favicon_64_PNG"));
            this->BUTTON_MINIMIZE->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Windows_App_Minimize_64_PNG"));
            this->BUTTON_MAXIMIZE->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Windows_App_Maximize_64_PNG"));
            this->BUTTON_CLOSE->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Windows_App_Close_64_PNG"));
        
            MyEA::File::Directory_Create("C:\\MyEA");

            MyEA::File::Directory_Create("C:\\MyEA\\LOG");
        }

        void CFormMain::OnPaint(PaintEventArgs^ e)
        {
        }

        void CFormMain::OnPaintBackground(PaintEventArgs^ e)
        {
        }

        /* Init_CE */
        const bool CFormMain::Init_CE(Common::CExpert* ptr_CE_received)
        {
            if(ptr_CE_received->ValidationSettings())
            {
                //this->Get__Ptr_CMyEA()->Push_Back(ptr_CE_received);

                // Initialize Title
                System::String^ tmp_ptr_title = gcnew System::String(ptr_CE_received->Get__Title().c_str());
                tmp_ptr_title += " - ";
                tmp_ptr_title += gcnew System::String(ptr_CE_received->Get__Ptr_CSymbolInfo()->Get__Name().c_str());
                this->FORM_TITLE_LABEL->Text = tmp_ptr_title;

                this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, gcnew System::String(ptr_CE_received->Get__Info().c_str()), 0xFFFFFFFF, true, true);

                return(true);
            } else { return(false); }
        }

        /* Maximized */
        void CFormMain::Maximized(void)
        {
            bool tmp_value = false;

            if(this->WindowState != FormWindowState::Maximized) { this->WindowState = FormWindowState::Maximized; }
            else
            {
                tmp_value = true;
                this->WindowState = FormWindowState::Normal;
            }

            TableLayoutRowStyleCollection^ tmp_ptr_row_styles = this->FormMainPrimaryTableLayoutPanel->RowStyles;
            RowStyle^ tmp_ptr_row_style = nullptr;
            for(int i(0); i < tmp_ptr_row_styles->Count; ++i)
            {
                if(i != 1)
                {
                    tmp_ptr_row_style = tmp_ptr_row_styles[i];
                    tmp_ptr_row_style->Height = (tmp_value ? 1.0F : 0.0F);
                }
            }
            tmp_ptr_row_style = nullptr;
            tmp_ptr_row_styles = nullptr;

            TableLayoutColumnStyleCollection^ tmp_ptr_column_styles = this->FormMainPrimaryTableLayoutPanel->ColumnStyles;
            ColumnStyle^ tmp_ptr_column_style = nullptr;
            for(int i(0); i < tmp_ptr_column_styles->Count; ++i)
            {
                if(i != 1)
                {
                    tmp_ptr_column_style = tmp_ptr_column_styles[i];
                    tmp_ptr_column_style->Width = (tmp_value ? 1.0F : 0.0F);
                }
            }
            tmp_ptr_column_style = nullptr;
            tmp_ptr_column_styles = nullptr;

            this->TopBorderPanel->Enabled = tmp_value;
            this->TopBorderPanel->Visible = tmp_value;
            this->LeftBorderPanel->Enabled = tmp_value;
            this->LeftBorderPanel->Visible = tmp_value;
            this->RightBorderPanel->Enabled = tmp_value;
            this->RightBorderPanel->Visible = tmp_value;
            this->BottomBorderPanel->Enabled = tmp_value;
            this->BottomBorderPanel->Visible = tmp_value;
        }

        void CFormMain::AutoResizeListView(ListView^ lisTview_received)
        {
            if(lisTview_received->Enabled && lisTview_received->Visible)
            {
                int tmp_count = lisTview_received->Columns->Count,
                    tmp_width_total = lisTview_received->Width;

                for(int i(0); i < tmp_count; ++i)
                {
                    lisTview_received->Columns[i]->Width = tmp_width_total/tmp_count;
                }
            }

            lisTview_received = nullptr;
        }

        void CFormMain::AutoResizeSignalFilterListView(void)
        {
            if(this->SignalFilterListView->Enabled && this->SignalFilterListView->Visible)
            {
                int tmp_count = this->SignalFilterListView->Columns->Count,
                    tmp_width_total = this->SignalFilterListView->Width;

                tmp_width_total -= 110;
                tmp_width_total -= 60;
                tmp_width_total -= 75*3;

                for(int i(0); i < tmp_count; ++i)
                {
                    if(i == 0) { this->SignalFilterListView->Columns[i]->Width = 110; } // Indicator Name
                    else if(i == 1) { this->SignalFilterListView->Columns[i]->Width = 60; } // Symbol Name
                    else if(i == 3 || // Period Bar(s)
                            i == 4 || // Condition(s)
                            i == 5) // Pattern
                    {
                        this->SignalFilterListView->Columns[i]->Width = 75;
                    }
                    else
                    {
                        // Time Frames
                        // Type Pattern
                        // Quality
                        this->SignalFilterListView->Columns[i]->Width = tmp_width_total/3;
                    }
                }
            }
        }

        /* Cin_TextBox */
        void CFormMain::Cin_TextBox(const Common::ENUM_TYPE_TEXTBOX tTextBox_received, System::String^ texTreceived)
        {
            switch(tTextBox_received)
            {
                case Common::TYPE_TB_NONE: break;
                default:     break;
            }

            delete(texTreceived);
        }

        /* Cin_RichTextBox */
        void CFormMain::Cin_RichTextBox(const Common::ENUM_TYPE_RICHTEXTBOX tRichTextBox_received, System::String^ ptr_texTreceived, unsigned int const hex_color_received, const bool append_texTreceived, const bool new_line_received)
        {
            RichTextBox^ tmp_ptr_rtb;

            switch(tRichTextBox_received) // Select The RichTextBox
            {
                case Common::TYPE_RTB_OUTPUT:
                    if(new_line_received) { ptr_texTreceived = ptr_texTreceived->Insert(0, gcnew System::String(MyEA::Time::Get__DateTimeFull().c_str())+" "); }
                    tmp_ptr_rtb = this->OutputRichTextBox;

                    if(tmp_ptr_rtb->Lines->Length > 1000)
                    {
                        tmp_ptr_rtb->Select(0, tmp_ptr_rtb->GetFirstCharIndexFromLine(tmp_ptr_rtb->Lines->Length - static_cast<int>(ceil(tmp_ptr_rtb->Lines->Length * 5.0f / 100.0f))));
                        tmp_ptr_rtb->ReadOnly = false;
                        tmp_ptr_rtb->SelectedText = "";
                        tmp_ptr_rtb->ReadOnly = true;
                    }
                        break;
                case Common::TYPE_RTB_STATUS_GENERAL: tmp_ptr_rtb = this->StatusGeneralRichTextBox; break;
                case Common::TYPE_RTB_STATUS_PRICE: tmp_ptr_rtb = this->StatusPriceRichTextBox; break;
                case Common::TYPE_RTB_STATUS_MONEY: tmp_ptr_rtb = this->StatusMoneyRichTextBox; break;
                case Common::TYPE_RTB_STATUS_NETWORK: tmp_ptr_rtb = this->StatusNetworkRichTextBox; break;
                case Common::TYPE_RTB_NONE:
                    tmp_ptr_rtb = this->StatusGeneralRichTextBox;

                    delete(ptr_texTreceived);
                    ptr_texTreceived = "ERROR";
                        break;
            }

            if(!append_texTreceived)  { tmp_ptr_rtb->Text = ""; } else if(new_line_received) { ptr_texTreceived = ptr_texTreceived->Insert(0, "\r\n"); }

            tmp_ptr_rtb->SelectionStart = tmp_ptr_rtb->TextLength;
            tmp_ptr_rtb->SelectionLength = 0;
            tmp_ptr_rtb->SelectionColor = Color::FromArgb(hex_color_received);
            tmp_ptr_rtb->AppendText(ptr_texTreceived);
            tmp_ptr_rtb->SelectionColor = tmp_ptr_rtb->ForeColor;

            // Scroll To The Bottom
            if(tRichTextBox_received == Common::TYPE_RTB_OUTPUT &&
               this->ActiveControl->Name != this->OutputRichTextBox->Name &&
               tmp_ptr_rtb->Enabled && tmp_ptr_rtb->Visible && !tmp_ptr_rtb->Focused)
            {
                /*
                tmp_ptr_rtb->SelectionStart = tmp_ptr_rtb->TextLength;
                tmp_ptr_rtb->ScrollToCaret();
                */
            }

            // Deinit
            delete(ptr_texTreceived);
        }

        void CFormMain::Cin_AccountInfo(System::String^ cin_texTreceived)
        {
            this->AccountInfo_Desc_Label->Text = cin_texTreceived;
            cin_texTreceived = cin_texTreceived->Insert(0, "[Account Info] ");
            this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, cin_texTreceived, 0xFFFDFFB8, true, true);

            delete(cin_texTreceived);
        }

        void CFormMain::Cin_PositionInfo(System::String^ cin_texTreceived)
        {
            this->PositionInfo_Desc_Label->Text = cin_texTreceived;
            cin_texTreceived = cin_texTreceived->Insert(0, "[Position Info] ");
            this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, cin_texTreceived, 0xFFFDFFB8, true, true);

            delete(cin_texTreceived);
        }

        void CFormMain::BUTTON_TRADE_TRADE_Click(System::Object^  sender, System::EventArgs^  e)
        {
            Trade_GoTo_Received(1);
        }

        void CFormMain::BUTTON_TRADE_HISTORY_Click(System::Object^  sender, System::EventArgs^  e)
        {
            Trade_GoTo_Received(2);
        }

        void CFormMain::BUTTON_TRADE_OPERATIONS_Click(System::Object^  sender, System::EventArgs^  e)
        {
            Trade_GoTo_Received(3);
        }

        void CFormMain::BUTTON_TRADE_RATIO_Click(System::Object^  sender, System::EventArgs^  e)
        {
            Trade_GoTo_Received(4);
        }

        /* Trade_GoTo_Received */
        void CFormMain::Trade_GoTo_Received(const unsigned short type_received)
        {
            TableLayoutColumnStyleCollection^ tmp_ptr_styles = this->TradeTableLayoutPanel->ColumnStyles;

            for(int i(1); i < tmp_ptr_styles->Count; ++i)
            {
                bool tmp_boolean(false);

                ColumnStyle^ tmp_ptr_row_style = tmp_ptr_styles[i];

                switch(i)
                {
                    case 1: // Trade
                        if(i == type_received) { tmp_boolean = true; }
                        this->EnableTableLayoutPanel(this->TradeTradeTableLayoutPanel, tmp_boolean);
                            break;
                    case 2: // History
                        if(i == type_received) { tmp_boolean = true; }
                        this->EnableTableLayoutPanel(this->TradeHistoryTableLayoutPanel, tmp_boolean);
                            break;
                    case 3: // Operations
                        if(i == type_received) { tmp_boolean = true; }
                        this->EnableTableLayoutPanel(this->TradeOperationsTableLayoutPanel, tmp_boolean);
                            break;
                    case 4: // Ratio
                        if(i == type_received) { tmp_boolean = true; }
                        this->EnableTableLayoutPanel(this->TradeRatioTableLayoutPanel, tmp_boolean);
                            break;
                }

                if(tmp_boolean)
                {
                    tmp_ptr_row_style->SizeType = SizeType::Percent;
                    tmp_ptr_row_style->Width = 100.0;
                }
                else
                {
                    tmp_ptr_row_style->SizeType = SizeType::Absolute;
                    tmp_ptr_row_style->Width = 0.0;
                }
            }
            tmp_ptr_styles = nullptr;

            if(type_received == 1) { this->AutoResizeListView(this->TradeTradeListView); }
            else if(type_received == 2) { this->AutoResizeListView(this->TradeHistoryListView); }
            else if(type_received == 3) { this->AutoResizeListView(this->TradeOperationsListView); }
            else if(type_received == 4)
            {
                this->AutoResizeListView(this->TradeRatioFloorListView);
                this->AutoResizeListView(this->TradeRatioRoundListView);
            }
        }

        void CFormMain::BUTTON_RATIO_FLOOR_CLICK(System::Object^  sender, System::EventArgs^  e)
        {
            this->Ratio_GoTo_Received(0);
        }

        void CFormMain::BUTTON_RATIO_ROUND_CLICK(System::Object^  sender, System::EventArgs^  e)
        {
            this->Ratio_GoTo_Received(1);
        }

        /* Ratio_GoTo_Received */
        void CFormMain::Ratio_GoTo_Received(const unsigned short type_received)
        {
            TableLayoutRowStyleCollection^ tmp_ptr_styles = this->TradeTypeRatioTableLayoutPanel->RowStyles;

            for(int i(0); i < tmp_ptr_styles->Count; ++i)
            {
                bool tmp_boolean(false);

                RowStyle^ tmp_ptr_row_style = tmp_ptr_styles[i];

                switch(i)
                {
                    case 0: // ListView Ratio Floor
                        if(i == type_received) { tmp_boolean = true; }
                        this->TradeRatioFloorListView->Enabled = tmp_boolean;
                        this->TradeRatioFloorListView->Visible = tmp_boolean;
                            break;
                    case 1: // ListView Ratio Round
                        if(i == type_received) { tmp_boolean = true; }
                        this->TradeRatioRoundListView->Enabled = tmp_boolean;
                        this->TradeRatioRoundListView->Visible = tmp_boolean;
                            break;
                }

                if(tmp_boolean)
                {
                    tmp_ptr_row_style->SizeType = SizeType::Percent;
                    tmp_ptr_row_style->Height = 100.0;
                } 
                else
                {
                    tmp_ptr_row_style->SizeType = SizeType::Absolute;
                    tmp_ptr_row_style->Height = 0.0;
                }
            }
            tmp_ptr_styles = nullptr;

            if(type_received == 0) { this->AutoResizeListView(this->TradeRatioFloorListView); }
            else if(type_received == 1) { this->AutoResizeListView(this->TradeRatioRoundListView); }
        }

        /* Main_GoTo_Received */
        void CFormMain::Main_GoTo_Received(Common::ENUM_TYPE_MAIN_PANEL tMainPanel_received)
        {
            bool tmp_boolean(false);

            TableLayoutRowStyleCollection^ tmp_ptr_styles = this->MainTableLayoutPanel->RowStyles;

            for(int i(0); i < tmp_ptr_styles->Count; ++i)
            {
                if(i > 0)
                {
                    tmp_boolean = false;

                    RowStyle^ tmp_ptr_row_style = tmp_ptr_styles[i];

                    switch((i+1))
                    {
                        case Common::TYPE_M_P_OUTPUT:
                            if((i+1) == tMainPanel_received) { tmp_boolean = true; }
                            this->EnableTableLayoutPanel(this->OutputTableLayoutPanel, tmp_boolean);
                                break;
                        case Common::TYPE_M_P_SIGNAL:
                            if((i+1) == tMainPanel_received) { tmp_boolean = true; }
                            this->EnableTableLayoutPanel(this->SignalTableLayoutPanel, tmp_boolean);
                            if(tmp_boolean)
                            {
                                this->AutoResizeSignalFilterListView();
                                this->AutoResizeListView(this->SignalListView);
                            }
                                break;
                        case Common::TYPE_M_P_ACCOUNT:
                            if((i+1) == tMainPanel_received) { tmp_boolean = true; }
                            this->EnableTableLayoutPanel(this->AccountTableLayoutPanel, tmp_boolean);
                                break;
                        case Common::TYPE_M_P_TRADE:
                            if((i+1) == tMainPanel_received) { tmp_boolean = true; }
                            this->EnableTableLayoutPanel(this->TradeTableLayoutPanel, tmp_boolean);
                            if(tmp_boolean)
                            {
                                this->AutoResizeListView(this->TradeTradeListView);
                                this->AutoResizeListView(this->TradeHistoryListView);
                                this->AutoResizeListView(this->TradeOperationsListView);
                                this->AutoResizeListView(this->TradeRatioFloorListView);
                                this->AutoResizeListView(this->TradeRatioRoundListView);
                            }
                                break;
                    }

                    if(tmp_boolean)
                    {
                        tmp_ptr_row_style->SizeType = SizeType::Percent;
                        tmp_ptr_row_style->Height = 100.0;
                    }
                    else
                    {
                        tmp_ptr_row_style->SizeType = SizeType::Absolute;
                        tmp_ptr_row_style->Height = 0.0;
                    }
                }

                if(3 == tMainPanel_received)
                {
                    this->AutoResizeSignalFilterListView();
                    this->AutoResizeListView(this->SignalListView);
                }
                else if(5 == tMainPanel_received)
                {
                    this->AutoResizeListView(this->TradeTradeListView);
                    this->AutoResizeListView(this->TradeHistoryListView);
                    this->AutoResizeListView(this->TradeOperationsListView);
                    this->AutoResizeListView(this->TradeRatioFloorListView);
                    this->AutoResizeListView(this->TradeRatioRoundListView);
                }
            }
            tmp_ptr_styles = nullptr;
        }


        void CFormMain::Signal_Modify(const double condition_received,
                                      unsigned int const filter_total_received,
                                      unsigned int const filter_total_low_received,
                                      unsigned int const filter_total_med_received,
                                      unsigned int const filter_total_high_received,
                                      System::String^ tool_tip_received,
                                      Common::ENUM_EVENTSIGNAL_WAIT event_signal_waiTreceived)
        {
            this->SignalListView->Items[0]->Text = (floor(condition_received * pow(10, 5)) / pow(10, 5)).ToString();
            this->SignalListView->Items[0]->ToolTipText = tool_tip_received->Replace("\t", "");
            this->SignalListView->Items[0]->SubItems[1]->Text = filter_total_received.ToString();
            this->SignalListView->Items[0]->SubItems[2]->Text = filter_total_low_received.ToString();
            this->SignalListView->Items[0]->SubItems[3]->Text = filter_total_med_received.ToString();
            this->SignalListView->Items[0]->SubItems[4]->Text = filter_total_high_received.ToString();
            this->SignalListView->Items[0]->SubItems[5]->Text = gcnew System::String(Common::ENUM_EVENTSIGNAL_WAIt_NAMES[event_signal_waiTreceived].c_str());
            std::string tmp_output = "";
            tmp_output += MyEA::Time::Get__DateTimeFull();
            tmp_output += " ";
            tmp_output += "[Signal]\r\n";
            this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, gcnew System::String(tmp_output.c_str())+tool_tip_received, 0xFF40BDBD, true, true);
        
            delete(tool_tip_received);
        }

        /* Signal_Filter_Ratio */
        const bool CFormMain::Signal_Filter_Ratio(const unsigned short type_received,
                                                                              const double condition_received,
                                                                              System::String^ symbol_received)
        {
            double tmp_condition(0.0);

            ListView^ tmp_lisTview = nullptr;

            for(int i(0); i < 2; ++i)
            {
                switch(i)
                {
                    case 0:
                        tmp_lisTview = this->TradeRatioFloorListView;
                        tmp_condition = floor(condition_received);
                            break;
                    case 1:
                        tmp_lisTview = this->TradeRatioRoundListView;
                        tmp_condition = round(condition_received);
                            break;
                }

                bool tmp_boolean(false);

                // Find it and modify
                for(int j(0); j < tmp_lisTview->Items->Count; ++j)
                {
                    if(tmp_lisTview->Items[j]->Text == symbol_received &&
                       tmp_lisTview->Items[j]->SubItems[1]->Text == tmp_condition.ToString())
                    {
                        System::String^ tmp_counTwin_string(tmp_lisTview->Items[j]->SubItems[2]->Text->ToString()->Substring(0, tmp_lisTview->Items[j]->SubItems[2]->Text->ToString()->IndexOf(" | ")));
                        System::String^    tmp_counTloss_string(tmp_lisTview->Items[j]->SubItems[3]->Text->ToString()->Substring(0, tmp_lisTview->Items[j]->SubItems[3]->Text->ToString()->IndexOf(" | ")));
                        System::String^    tmp_counTdrawn_string(tmp_lisTview->Items[j]->SubItems[4]->Text->ToString()->Substring(0, tmp_lisTview->Items[j]->SubItems[4]->Text->ToString()->IndexOf(" | ")));

                        unsigned long long tmp_counTwin((type_received == 2 ? 1 : 0)+UInt64::Parse(tmp_counTwin_string)),
                                                        tmp_counTloss((type_received == 1 ? 1 : 0)+UInt64::Parse(tmp_counTloss_string)),
                                                        tmp_counTdrawn((type_received == 0 ? 1 : 0)+UInt64::Parse(tmp_counTdrawn_string)),
                                                        tmp_counTtrade(tmp_counTwin+tmp_counTloss+tmp_counTdrawn);
                    
                        tmp_lisTview->Items[j]->SubItems[2]->Text = (tmp_counTwin).ToString()+" | "+(tmp_counTwin*100/tmp_counTtrade).ToString()+"%"; // Count Win
                        tmp_lisTview->Items[j]->SubItems[3]->Text =(tmp_counTloss).ToString()+" | "+(tmp_counTloss*100/tmp_counTtrade).ToString()+"%"; // Count Loss
                        tmp_lisTview->Items[j]->SubItems[4]->Text = (tmp_counTdrawn).ToString()+" | "+(tmp_counTdrawn*100/tmp_counTtrade).ToString()+"%"; // Count Drawn
                    
                        tmp_boolean = true;
                            break;
                    }
                }

                // Not Find, Add It To The ListView
                if(!tmp_boolean)
                {
                    ListViewItem^ tmp_lvi = gcnew ListViewItem();
                    tmp_lvi->ForeColor = Color::White;
                    tmp_lvi->Text = symbol_received;
                    tmp_lvi->SubItems->Add(tmp_condition.ToString());
                    tmp_lvi->SubItems->Add((type_received == 2 ? 1 : 0).ToString()+" | "+(type_received == 2 ? "100%" : "0%"));
                    tmp_lvi->SubItems->Add((type_received == 1 ? 1 : 0).ToString()+" | "+(type_received == 1 ? "100%" : "0%"));
                    tmp_lvi->SubItems->Add((type_received == 0 ? 1 : 0).ToString()+" | "+(type_received == 0 ? "100%" : "0%"));
                    tmp_lisTview->Items->Insert(0, tmp_lvi);
                }
            }

            tmp_lisTview = nullptr;
            delete(symbol_received);

            return(true);
        }

        /* Signal_Filter_Push_Back */
        const bool CFormMain::Signal_Filter_Push_Back(const Common::ENUM_TYPE_INDICATORS tIndicator_received,
                                                                                      System::String^ symbol_received,
                                                                                      System::String^ period_bars_received,
                                                                                      System::String^ tool_tip_received,
                                                                                      Common::ENUM_TIME_FRAMES time_frames_received)
        {
            bool tmp_boolean(true);

            this->p_Ptr_l_indicator->Insert(0, tIndicator_received);

            ListViewItem^ tmp_lvi = gcnew ListViewItem();
            tmp_lvi->ForeColor = Color::White;
            tmp_lvi->ToolTipText = tool_tip_received->Replace("\t", "");
            tmp_lvi->SubItems->Add(symbol_received);
            tmp_lvi->SubItems->Add(gcnew System::String(Common::ENUM_TIME_FRAMES_NAMES[time_frames_received].c_str()));
            tmp_lvi->SubItems->Add(period_bars_received);
            tmp_lvi->SubItems->Add("0");
            tmp_lvi->SubItems->Add("-1");
            tmp_lvi->SubItems->Add("TYPE_PATTERN_NONE");
            tmp_lvi->SubItems->Add("TYPE_SQ_NONE");
            switch(tIndicator_received)
            {
                case Common::TYPE_iNONE: tmp_lvi->Text  = "None"; break;
                case Common::TYPE_iNONE_NN: tmp_lvi->Text  = "None NN"; break;
                case Common::TYPE_iNONE_RNN: tmp_lvi->Text  = "None RNN"; break;
                case Common::TYPE_iBEARSPOWER: tmp_lvi->Text  = "Bears Power"; break;
                case Common::TYPE_iBEARSPOWER_NN: tmp_lvi->Text  = "Bears Power NN"; break;
                case Common::TYPE_iBULLSPOWER: tmp_lvi->Text  = "Bulls Power"; break;
                case Common::TYPE_iBULLSPOWER_NN: tmp_lvi->Text  = "Bulls Power NN"; break;
                case Common::TYPE_iCANDLESTICK: tmp_lvi->Text  = "CandleStick NN"; break;
                case Common::TYPE_iCCI: tmp_lvi->Text  = "CCI"; break;
                case Common::TYPE_iCCI_NN: tmp_lvi->Text  = "CCI NN"; break;
                case Common::TYPE_iFIBONACCI: tmp_lvi->Text  = "Fibonacci"; break;
                case Common::TYPE_iFIBONACCI_NN: tmp_lvi->Text  = "Fibonacci NN"; break;
                case Common::TYPE_iICHIMOKUKINKOHYO: tmp_lvi->Text  = "Ichimoku Kinko Hyo"; break;
                case Common::TYPE_iMACD: tmp_lvi->Text = "MACD"; break;
                case Common::TYPE_iMACD_NN: tmp_lvi->Text = "MACD NN"; break;
                case Common::TYPE_iMA: tmp_lvi->Text  = "MA"; break;
                case Common::TYPE_iMA_RNN_SIGN: tmp_lvi->Text  = "MA RNN"; break;
                case Common::TYPE_iMA_RNN_PRICE: tmp_lvi->Text  = "MA RNN"; break;
                case Common::TYPE_iRSI: tmp_lvi->Text  = "RSI"; break;
                case Common::TYPE_iRSI_NN: tmp_lvi->Text  = "RSI NN"; break;
                case Common::TYPE_iRVI: tmp_lvi->Text  = "RVI"; break;
                case Common::TYPE_iRVI_NN: tmp_lvi->Text  = "RVI NN"; break;
                case Common::TYPE_iSTOCHASTIC: tmp_lvi->Text  = "Stochastic"; break;
                case Common::TYPE_iSTOCHASTIC_NN: tmp_lvi->Text  = "Stochastic NN"; break;
                default: tmp_boolean = false; break;
            }

            if(tmp_boolean)  { this->SignalFilterListView->Items->Add(tmp_lvi); }
            else { delete(tmp_lvi); }

            delete(symbol_received);
            delete(period_bars_received);
            delete(tool_tip_received);

            return(tmp_boolean);
        }

        /* Signal_Filter_Modify */
        const bool CFormMain::Signal_Filter_Modify(const Common::ENUM_TYPE_INDICATORS tIndicator_received,
                                                                               const int condition_received,
                                                                               const int pattern_received,
                                                                               const Common::ENUM_TYPE_SIGNAL_PATTERN tSignal_Pattern_received,
                                                                               const Common::ENUM_TYPE_SIGNAL_QUALITY tSignal_Quality_received,
                                                                               System::String^ symbol_received,
                                                                               System::String^ period_bars_received,
                                                                               System::String^ tool_tip_received,
                                                                               Common::ENUM_TIME_FRAMES time_frames_received)
        {
            bool tmp_boolean(false);

            for(int i(0); i < this->SignalFilterListView->Items->Count; ++i)
            {
                if(tIndicator_received == this->p_Ptr_l_indicator[i] &&
                   this->SignalFilterListView->Items[i]->SubItems[1]->Text == symbol_received &&
                   this->SignalFilterListView->Items[i]->SubItems[2]->Text == gcnew System::String(Common::ENUM_TIME_FRAMES_NAMES[time_frames_received].c_str()) &&
                   this->SignalFilterListView->Items[i]->SubItems[3]->Text == period_bars_received)
                {
                    this->SignalFilterListView->Items[i]->ToolTipText = tool_tip_received->Replace("\t", "");
                    this->SignalFilterListView->Items[i]->SubItems[4]->Text = condition_received.ToString();
                    this->SignalFilterListView->Items[i]->SubItems[5]->Text = pattern_received.ToString();
                    this->SignalFilterListView->Items[i]->SubItems[6]->Text = gcnew System::String(Common::ENUM_TYPE_SIGNAL_PATTERN_NAMES[tSignal_Pattern_received].c_str());
                    this->SignalFilterListView->Items[i]->SubItems[7]->Text = gcnew System::String(Common::ENUM_TYPE_SIGNAL_QUALITY_NAMES[tSignal_Quality_received].c_str());
                    std::string tmp_output = "";
                    tmp_output += MyEA::Time::Get__DateTimeFull();
                    tmp_output += " ";
                    switch(tIndicator_received)
                    {
                        case Common::TYPE_iNONE: tmp_output += "[Signal None]\r\n"; break;
                        case Common::TYPE_iNONE_NN: tmp_output += "[Signal None NN]\r\n"; break;
                        case Common::TYPE_iNONE_RNN: tmp_output += "[Signal None RNN]\r\n"; break;
                        case Common::TYPE_iBEARSPOWER: tmp_output += "[Signal Bears Power]\r\n"; break;
                        case Common::TYPE_iBEARSPOWER_NN: tmp_output += "[Signal Bears Power NN]\r\n"; break;
                        case Common::TYPE_iBULLSPOWER: tmp_output += "[Signal Bulls Power]\r\n"; break;
                        case Common::TYPE_iBULLSPOWER_NN: tmp_output += "[Signal Bulls Power NN]\r\n"; break;
                        case Common::TYPE_iCANDLESTICK: tmp_output += "[Signal CandleStick]\r\n"; break;
                        case Common::TYPE_iCCI: tmp_output += "[Signal CCI]\r\n"; break;
                        case Common::TYPE_iCCI_NN: tmp_output += "[Signal CCI NN]\r\n"; break;
                        case Common::TYPE_iFIBONACCI: tmp_output += "[Signal Fibonacci]\r\n"; break;
                        case Common::TYPE_iFIBONACCI_NN: tmp_output += "[Signal Fibonacci NN]\r\n"; break;
                        case Common::TYPE_iICHIMOKUKINKOHYO: tmp_output += "[Signal Ichimoku Kinko Hyo]"; break;
                        case Common::TYPE_iMACD: tmp_output += "[Signal MACD]\r\n"; break;
                        case Common::TYPE_iMACD_NN: tmp_output += "[Signal MACD NN]\r\n"; break;
                        case Common::TYPE_iMA: tmp_output += "[Signal MA]\r\n"; break;
                        case Common::TYPE_iMA_RNN_SIGN: tmp_output += "[Signal MA RNN]\r\n"; break;
                        case Common::TYPE_iMA_RNN_PRICE: tmp_output += "[Signal MA RNN]\r\n"; break;
                        case Common::TYPE_iRSI: tmp_output += "[Signal RSI]\r\n"; break;
                        case Common::TYPE_iRSI_NN: tmp_output += "[Signal RSI NN]\r\n"; break;
                        case Common::TYPE_iRVI: tmp_output += "[Signal RVI]\r\n"; break;
                        case Common::TYPE_iRVI_NN: tmp_output += "[Signal RVI NN]\r\n"; break;
                        case Common::TYPE_iSTOCHASTIC: tmp_output += "[Signal Stochastic]\r\n"; break;
                        case Common::TYPE_iSTOCHASTIC_NN: tmp_output += "[Signal Stochastic NN]\r\n"; break;
                        default: tmp_output += "[Signal Unknow]\r\n"; break;
                    }

                    this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, gcnew System::String(tmp_output.c_str())+tool_tip_received, 0xFF40BDBD, true, true);
                
                    tmp_boolean = true;
                    break;
                }
            }

            delete(symbol_received);
            delete(period_bars_received);
            delete(tool_tip_received);

            return(tmp_boolean);
        }

        /* History_UnShift */
        const bool CFormMain::History_UnShift(const Common::ENUM_TYPE_POSITION tPosition_received,
                                                                         const unsigned long long order_received,
                                                                         const double price_received,
                                                                         const double stop_level_received,
                                                                         const double take_level_received,
                                                                         System::String^ symbol_received,
                                                                         System::String^ date_0_received,
                                                                         System::String^ date_1_received,
                                                                         System::String^ comment_received,
                                                                         System::String^ tool_tip_received)
        {
            ListViewItem^ tmp_lvi = gcnew ListViewItem();
            tmp_lvi->ForeColor = Color::White;
            switch(tPosition_received)
            {
                case Common::TYPE_POSITION_BUY: tmp_lvi->Text = "B : "; break;
                case Common::TYPE_POSITION_SELL: tmp_lvi->Text = "S : "; break;
                case Common::TYPE_POSITION_NONE:
                default: tmp_lvi->Text = "NONE : ";  break;
            }
            tmp_lvi->Text += date_0_received;
            tmp_lvi->ToolTipText = tool_tip_received->Replace("\t", "");
            tmp_lvi->SubItems->Add(order_received.ToString());
            tmp_lvi->SubItems->Add(symbol_received);
            tmp_lvi->SubItems->Add("~");
            tmp_lvi->SubItems->Add("~");
            tmp_lvi->SubItems->Add((floor(price_received*pow(10, 5))/pow(10, 5)).ToString());
            tmp_lvi->SubItems->Add((floor(stop_level_received*pow(10, 5))/pow(10, 5)).ToString());
            tmp_lvi->SubItems->Add((floor(take_level_received*pow(10, 5))/pow(10, 5)).ToString());
            tmp_lvi->SubItems->Add(date_1_received);
            tmp_lvi->SubItems->Add("~");
            tmp_lvi->SubItems->Add(comment_received);
            this->TradeHistoryListView->Items->Insert(0, tmp_lvi);

            tmp_lvi = nullptr;
            delete(symbol_received);
            delete(date_0_received);
            delete(date_1_received);
            delete(comment_received);
            delete(tool_tip_received);

            return(true);
        }

        /* EnableTableLayoutPanel */
        void CFormMain::EnableTableLayoutPanel(TableLayoutPanel^ tlp_received, const bool enable_received)
        {
            for(int i(0); i < tlp_received->ColumnCount; ++i)
            {
                for(int j = 0; j < tlp_received->RowCount; ++j)
                {
                    if(tlp_received->GetControlFromPosition(i, j) != nullptr)
                    {
                        tlp_received->GetControlFromPosition(i, j)->Enabled = enable_received;
                        tlp_received->GetControlFromPosition(i, j)->Visible = enable_received;
                    }
                }
            }

            tlp_received->Enabled = enable_received;
            tlp_received->Visible = enable_received;
        }

        /* Main_GoTo_Init_ */
        const bool CFormMain::Main_GoTo_Init_(void)
        {
            TableLayoutRowStyleCollection^ tmp_ptr_styles = this->MainTableLayoutPanel->RowStyles;
            for(int i(0); i < tmp_ptr_styles->Count; ++i)
            {
                if(i > 0)
                {
                    RowStyle^ tmp_ptr_row_style = tmp_ptr_styles[i];
                    if(tmp_ptr_row_style->SizeType == SizeType::Percent)
                    {
                        this->p_type_main_panel = static_cast<Common::ENUM_TYPE_MAIN_PANEL>(i+1);
                        this->Main_GoTo_Received(p_type_main_panel);
                    }
                }
            }
            tmp_ptr_styles = nullptr;

            return(false);
        }

        /* FormMain_Load */
        void CFormMain::FormMain_Load(Object^  sender, EventArgs^  e)
        {
            this->ActiveControl = this->LOGO;
            this->_moving = false;
            this->_sizing = false;

            Main_GoTo_Init_();

            System::String^ tmp_ptr_sVersion = "?";
            this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, tmp_ptr_sVersion, 0xFFFFFFFF, false, false);
            this->Cin_RichTextBox(Common::TYPE_RTB_STATUS_GENERAL, "Ready", 0xFFFFFFFF, false, false);

            delete(tmp_ptr_sVersion);
        }

        void CFormMain::FormMain_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->WindowState != FormWindowState::Maximized)
            {
                this->_moving = true;
                this->_position_offset = Point(e->X, e->Y);
            }
        }
        void CFormMain::FormMain_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->_moving)
            {
                Point tmp_current_screen_position = PointToScreen(e->Location);
                this->Location = Point(tmp_current_screen_position.X - this->_position_offset.X, tmp_current_screen_position.Y - this->_position_offset.Y);
            }
        }
        void CFormMain::FormMain_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_moving = false;
        }

        // BUTTON_CLOSE
        void CFormMain::BUTTON_CLOSE_Click(System::Object^  sender, System::EventArgs^  e) { Exit(false); }

        // BUTTON_MAXIMIZE
        void CFormMain::BUTTON_MAXIMIZE_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Maximized();
        }

        // BUTTON_MINIMIZE
        void CFormMain::BUTTON_MINIMIZE_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->WindowState = FormWindowState::Minimized;
        }

        // BUTTON_EXIT
        void CFormMain::BUTTON_FILE_Click(System::Object^  sender, System::EventArgs^  e) { Exit(false); }

        void CFormMain::Exit(const bool exiTprocess_received)
        {
            this->isThreading = false;

            if(exiTprocess_received) { exit(0); }
            else { Close(); }
        }

        // Maximized_DoubleClick
        void CFormMain::Maximized_DoubleClick(System::Object^  sender, System::EventArgs^  e)
        {
            this->Maximized();
        }

        // RightBorderPanel
        void CFormMain::FormMain_ReSize_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_sizing = true;
            POINT tmp_cursor_position;
            GetCursorPos(&tmp_cursor_position);
            this->_size_offset = Point((int)tmp_cursor_position.x, (int)tmp_cursor_position.y);
        }
        System::Void CFormMain::FormMain_ReSize_MouseMove_Top(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_position_offset = Point(e->X, e->Y);
            ReSize(0);
        }
        System::Void CFormMain::FormMain_ReSize_MouseMove_Left(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_position_offset = Point(e->X, e->Y);
            ReSize(1);
        }
        System::Void CFormMain::FormMain_ReSize_MouseMove_Right(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            ReSize(2);
        }
        System::Void CFormMain::FormMain_ReSize_MouseMove_Bottom(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            ReSize(3);
        }
        System::Void CFormMain::ReSize(const unsigned short id_received)
        {
            if(this->_sizing)
            {
                POINT tmp_cursor_position;
                GetCursorPos(&tmp_cursor_position);
                int tmp_size = 0;
                switch(id_received)
                {
                    case 0: // Top
                        tmp_size = (int)tmp_cursor_position.y-_size_offset.Y;
                        if((tmp_size > 0 && this->Size.Height > (int)p_min_form_height) ||
                           tmp_size < 0)
                        {
                            this->Location = Point(this->Location.X, (int)tmp_cursor_position.y+tmp_size);
                            this->Size = System::Drawing::Size(this->Size.Width, this->Size.Height-tmp_size);
                            //this->_size_offset = Point(this->_size_offset.X, (int)tmp_cursor_position.y);
                        }
                        break;
                    case 1: // Left
                        tmp_size = (int)tmp_cursor_position.x-_size_offset.X;
                        if((tmp_size > 0 && this->Size.Width > (int)p_min_form_width) ||
                           tmp_size < 0)
                        {
                            this->Location = Point((int)tmp_cursor_position.x+tmp_size, this->Location.Y);
                            this->Size = System::Drawing::Size(this->Size.Width-tmp_size, this->Size.Height);
                            this->_size_offset = Point((int)tmp_cursor_position.x, this->_size_offset.Y);
                        }
                        break;
                    case 2: // Right
                        tmp_size = (int)tmp_cursor_position.x-_size_offset.X;
                        if(tmp_size > 0 ||
                           (tmp_size < 0 && this->Size.Width >(int)p_min_form_width))
                        {
                            this->Size = System::Drawing::Size(this->Size.Width+tmp_size, this->Size.Height);
                            this->_size_offset = Point((int)tmp_cursor_position.x, this->_size_offset.Y);
                        }
                        break;
                    case 3: // Bottom
                        tmp_size = (int)tmp_cursor_position.y-_size_offset.Y;
                        if(tmp_size > 0 ||
                           (tmp_size < 0 && this->Size.Height >(int)p_min_form_height))
                        {
                            this->Size = System::Drawing::Size(this->Size.Width, this->Size.Height+tmp_size);
                            this->_size_offset = Point(this->_size_offset.X, (int)tmp_cursor_position.y);
                        }
                        break;
                }
            }
        }
        void CFormMain::FormMain_ReSize_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_sizing = false;
        }

        // LogTextBox
        void CFormMain::OutputRichTextBox_KeyPress(Object^  sender, KeyPressEventArgs^  e)
        {
            this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, "Press -> "+e->KeyChar, 0xFFFFFFFF, true, true);
            if(e->KeyChar == '^C')
            {
                this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, " CONTROL KEY + C", 0xFFFFFFFF, true, true);
                Clipboard::SetDataObject(this->OutputRichTextBox->SelectedText, true);
            }
        }

        const bool CFormMain::FormLogin(void)
        {
            CFormLogin^ tmp_ptr_CFL(gcnew CFormLogin());
            tmp_ptr_CFL->ShowDialog();
            
            bool tmp_boolean(tmp_ptr_CFL->Get__Connected());
            
            this->Cin_RichTextBox(Common::TYPE_RTB_STATUS_NETWORK, tmp_boolean ? "ONLINE" : "OFFLINE", 0xFFFFFFFF, false, false);
            
            delete(tmp_ptr_CFL);
            return(tmp_boolean);
        }

        // BUTTON_VIEW
        void CFormMain::BUTTON_VIEW_Click(Object^  sender, EventArgs^  e)
        {
            this->Cin_RichTextBox(Common::TYPE_RTB_OUTPUT, "BUTTON_VIEW_Click", 0xFFFFFFFF, true, true);
        }

        // BUTTON_MENU_OUTPUT
        void CFormMain::BUTTON_MENU_OUTPUTClick(Object^  sender, EventArgs^  e) { this->Main_GoTo_Received(Common::TYPE_M_P_OUTPUT); }

        // BUTTON_MENU_SIGNAL
        void CFormMain::BUTTON_MENU_SIGNAL_Click(Object^  sender, EventArgs^  e) { this->Main_GoTo_Received(Common::TYPE_M_P_SIGNAL); }

        // BUTTON_MENU_ACCOUNT
        void CFormMain::BUTTON_MENU_ACCOUNTClick(Object^  sender, EventArgs^  e) { this->Main_GoTo_Received(Common::TYPE_M_P_ACCOUNT); }

        // BUTTON_MENU_TRADE
        void CFormMain::BUTTON_MENU_TRADE_Click(Object^  sender, EventArgs^  e) { this->Main_GoTo_Received(Common::TYPE_M_P_TRADE); }

        void CFormMain::AboutMyEAToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
        {
            p_Ptr_CFA = gcnew CFormAbout(p_Ptr_Ressource_Manager);
            p_Ptr_CFA->ShowDialog();
        }

        CFormMain::~CFormMain(void)
        {
            if(this->p_Ptr_l_indicator) { delete(this->p_Ptr_l_indicator); }
            if(this->p_Ptr_CFA) { delete(this->p_Ptr_CFA); }
            if(this->p_Ptr_Ressource_Manager) { delete(this->p_Ptr_Ressource_Manager); }
            if(this->components) { delete(this->components); }
        }
    }
}