#include "stdafx.hpp"

#include <Form/CFormNN.h>

#include <cmath>

namespace MyEA
{
    namespace Form
    {
        /* MAIN_CFNN */
        const int MAIN_CFNN(CFormNN^ ptr_CFNN_received)
        {
            // Enabling Windows XP visual effects before any controls are created
            Application::EnableVisualStyles();
            Application::SetCompatibleTextRenderingDefault(false);

            Application::Run(ptr_CFNN_received);

            return(0);
        }

        CFormNN::CFormNN(void) : isThreading(true),
                                                    _moving(false),
                                                    _counTindex(0),
                                                    _position_offset(0, 0),
                                                    p_Ptr_Ressource_Manager(gcnew ResourceManager("FormWin.Resource_Files.Resource", GetType()->Assembly))
        {
            InitializeComponent();
            //
            //TODO: Add the constructor code here
            //

            this->Icon = safe_cast<System::Drawing::Icon^>(p_Ptr_Ressource_Manager->GetObject("COMPANY_favicon_64_ICO"));
            this->NN_BUTTON_CLOSE->BackgroundImage = safe_cast<Image^>(p_Ptr_Ressource_Manager->GetObject("Windows_App_Close_64_PNG"));
        }

        /* FormNN_Load */
        void CFormNN::FormNN_Load(Object^  sender, EventArgs^  e)
        {
            this->Parameter_Reset();
        }

        const bool CFormNN::Get__Parameter_Init_(void) { return(this->_parameter_init); }

        void CFormNN::Maximized(void)
        {

            if(this->WindowState != FormWindowState::Maximized) { this->WindowState = FormWindowState::Maximized; }
            else { this->WindowState = FormWindowState::Normal; }
        }

        void CFormNN::NNTopTableLayoutPanel_DoubleClick(System::Object^  sender, System::EventArgs^  e)
        {
            this->Maximized();
        }

        void CFormNN::FormNN_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->WindowState != FormWindowState::Maximized)
            {
                this->_moving = true;
                this->_position_offset = Point(e->X, e->Y);
            }
        }
        void CFormNN::FormNN_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            if(this->_moving)
            {
                Point currentScreenPos = PointToScreen(e->Location);
                this->Location = Point(currentScreenPos.X - this->_position_offset.X, currentScreenPos.Y - this->_position_offset.Y);
            }
        }
        void CFormNN::FormNN_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
        {
            this->_moving = false;
        }

        void CFormNN::FormClose(void)
        {
            this->Close();
        }

        void CFormNN::NN_BUTTON_CLOSE_Click(Object^  sender, EventArgs^  e)
        {
            this->FormClose();
        }

        void CFormNN::Cin_NN_Info(System::String^ cin_texTreceived)
        {
            this->INFO_LABEL->Text = cin_texTreceived;
            
            delete(cin_texTreceived);
        }

        delegate void Chart_MSE_Init_Delegate(const bool is_long_type_received, unsigned int const counTseries_received, const double counTinterval_datapoint);
        void CFormNN::Chart_MSE_Init_(const bool is_long_type_received, unsigned int const counTseries_received, const double counTinterval_datapoint)
        {
            if(this->InvokeRequired)
            {
                Chart_MSE_Init_Delegate^ tmp_invoke_action = gcnew Chart_MSE_Init_Delegate(this, &CFormNN::Chart_MSE_Init_);
                
                this->Invoke(tmp_invoke_action, is_long_type_received, counTseries_received, counTinterval_datapoint);
                
                delete(tmp_invoke_action);

                return;
            }

            this->MSE_CHART->ChartAreas["MSE_ChartArea"]->AxisX->Interval = counTinterval_datapoint - 1;

            System::String^ tmp_ptr_String_besTtest = gcnew System::String("MSE_Series_BesTTesT"+(is_long_type_received ? "Long" : "Short")+"_");
            System::String^ tmp_ptr_String_besTtrain = gcnew System::String("MSE_Series_BesTTrain_" + (is_long_type_received ? "Long" : "Short") + "_");
            System::String^ tmp_ptr_String_trainer_test = gcnew System::String("MSE_Series_Trained_TesT"+(is_long_type_received ? "Long" : "Short")+"_");
            System::String^ tmp_ptr_String_trainer_train = gcnew System::String("MSE_Series_Trained_Train_"+(is_long_type_received ? "Long" : "Short")+"_");
            
            for(unsigned int i(0); i < counTseries_received; ++i)
            {
                Series^ tmp_series_besTtest = gcnew Series;

                tmp_series_besTtest->Name = tmp_ptr_String_besTtest + i.ToString();
                tmp_series_besTtest->Color = is_long_type_received ? Color::Turquoise : Color::Orange;
                tmp_series_besTtest->ChartType = SeriesChartType::Line;

                this->MSE_CHART->Series->Add(tmp_series_besTtest);

                Series^ tmp_series_besTtrain = gcnew Series;

                tmp_series_besTtrain->Name = tmp_ptr_String_besTtrain + i.ToString();
                tmp_series_besTtrain->Color = is_long_type_received ? Color::Violet : Color::DarkOrange;
                tmp_series_besTtrain->ChartType = SeriesChartType::Line;

                this->MSE_CHART->Series->Add(tmp_series_besTtrain);

                Series^ tmp_series_trainer_test = gcnew Series;

                tmp_series_trainer_test->Name = tmp_ptr_String_trainer_test + i.ToString();
                tmp_series_trainer_test->Color = is_long_type_received ? Color::CadetBlue : Color::Salmon;
                tmp_series_trainer_test->ChartType = SeriesChartType::Line;

                this->MSE_CHART->Series->Add(tmp_series_trainer_test);

                Series^ tmp_series_trainer_train = gcnew Series;

                tmp_series_trainer_train->Name = tmp_ptr_String_trainer_train + i.ToString();
                tmp_series_trainer_train->Color = is_long_type_received ? Color::Blue : Color::Red;
                tmp_series_trainer_train->ChartType = SeriesChartType::Line;

                this->MSE_CHART->Series->Add(tmp_series_trainer_train);
            }

            delete(tmp_ptr_String_besTtest);
            delete(tmp_ptr_String_besTtrain);
            delete(tmp_ptr_String_trainer_test);
            delete(tmp_ptr_String_trainer_train);

            this->_counTindex = counTseries_received;
        }

        delegate void Chart_MSE_Interval_Delegate(const double counTinterval_datapoint);
        void CFormNN::Chart_MSE_Interval(const double counTinterval_datapoint)
        {
            if(this->InvokeRequired)
            {
                Chart_MSE_Interval_Delegate^ tmp_invoke_action = gcnew Chart_MSE_Interval_Delegate(this, &CFormNN::Chart_MSE_Interval);
                
                this->Invoke(tmp_invoke_action, counTinterval_datapoint);
                
                delete(tmp_invoke_action);

                return;
            }

            this->MSE_CHART->ChartAreas["MSE_ChartArea"]->AxisX->Interval = counTinterval_datapoint - 1;
        }

        delegate void Chart_MSE_Set_Info_Delegate(const unsigned short type_received, System::String^ texTreceived);
        System::Void CFormNN::Chart_MSE_Set_Info(const unsigned short type_received, System::String^ texTreceived)
        {
            if(this->InvokeRequired)
            {
                Chart_MSE_Set_Info_Delegate^ tmp_invoke_action = gcnew Chart_MSE_Set_Info_Delegate(this, &CFormNN::Chart_MSE_Set_Info);
                
                this->Invoke(tmp_invoke_action, type_received, texTreceived);
                
                delete(tmp_invoke_action);

                return;
            }

            switch(type_received)
            {
                case 0: this->MSE_Current_Epoch_Done_Label->Text = texTreceived; break;
                case 1: this->MSE_Total_Epoch_Done_Label->Text = texTreceived; break;
                case 2: this->MSE_Current_Label->Text = texTreceived; break;
                case 3: this->MSE_Higher_Label->Text = texTreceived; break;
                case 4: this->MSE_Lower_Label->Text = texTreceived; break;
                case 5: this->MSE_Higher_High_Label->Text = texTreceived; break;
                case 6: this->MSE_Lower_Low_Label->Text = texTreceived; break;
                case 7: this->MSE_ANNS_TRAINING_Label->Text = texTreceived; break;
            }
        }

        delegate void Chart_MSE_Set_Title_X_Delegate(System::String^ title_X_received);
        System::Void CFormNN::Chart_MSE_Set_Title_X(System::String^ title_X_received)
        {
            if(this->InvokeRequired)
            {
                Chart_MSE_Set_Title_X_Delegate^ tmp_invoke_action = gcnew Chart_MSE_Set_Title_X_Delegate(this, &CFormNN::Chart_MSE_Set_Title_X);
                
                this->Invoke(tmp_invoke_action, title_X_received);
                
                delete(tmp_invoke_action);

                return;
            }

            this->MSE_CHART->ChartAreas["MSE_ChartArea"]->AxisX->Title = title_X_received;
        }
        
        delegate void Chart_MSE_Reset__PoinTDelegate(const bool is_long_type_received);
        void CFormNN::Chart_MSE_Reset__Point(const bool is_long_type_received)
        {
            if(this->InvokeRequired)
            {
                Chart_MSE_Reset__PoinTDelegate^ tmp_invoke_action = gcnew Chart_MSE_Reset__PoinTDelegate(this, &CFormNN::Chart_MSE_Reset__Point);
                
                this->Invoke(tmp_invoke_action, is_long_type_received);
                
                delete(tmp_invoke_action);

                return;
            }

            System::String^ tmp_ptr_String_besTtest = gcnew System::String("MSE_Series_BesTTesT" + (is_long_type_received ? "Long" : "Short") + "_");
            System::String^ tmp_ptr_String_besTtrain = gcnew System::String("MSE_Series_BesTTrain_" + (is_long_type_received ? "Long" : "Short") + "_");
            System::String^ tmp_ptr_String_trainer_test = gcnew System::String("MSE_Series_Trained_TesT" + (is_long_type_received ? "Long" : "Short") + "_");
            System::String^ tmp_ptr_String_trainer_train = gcnew System::String("MSE_Series_Trained_Train_" + (is_long_type_received ? "Long" : "Short") + "_");
            for(unsigned int i(0); i < this->_counTindex; ++i)
            {
                this->MSE_CHART->Series[tmp_ptr_String_besTtest+ i.ToString()]->Points->Clear();
                this->MSE_CHART->Series[tmp_ptr_String_besTtrain + i.ToString()]->Points->Clear();
                this->MSE_CHART->Series[tmp_ptr_String_trainer_test + i.ToString()]->Points->Clear();
                this->MSE_CHART->Series[tmp_ptr_String_trainer_train + i.ToString()]->Points->Clear();
            }
        }

        delegate void Chart_MSE_Add_PoinTDelegate(const unsigned char type_ann_use_received,
                                                                                    const bool is_long_type_received,
                                                                                    const bool is_tesTtype_received,
                                                                                    unsigned int const ann_received,
                                                                                    const double x_received,
                                                                                    const double y_received);
        void CFormNN::Chart_MSE_Add_Point(const unsigned char type_ann_use_received,
                                                                        const bool is_long_type_received,
                                                                        const bool is_tesTtype_received,
                                                                        unsigned int const ann_received,
                                                                        const double x_received,
                                                                        const double y_received)
        {
            if(this->InvokeRequired)
            {
                Chart_MSE_Add_PoinTDelegate^ tmp_invoke_action = gcnew Chart_MSE_Add_PoinTDelegate(this, &CFormNN::Chart_MSE_Add_Point);
                
                this->Invoke(tmp_invoke_action, type_ann_use_received, is_long_type_received, is_tesTtype_received, ann_received, x_received, y_received);
                
                delete(tmp_invoke_action);

                return;
            }

            Common::ENUM_TYPE_NEURAL_NETWORK_USE tmp_type_ann_use_received(static_cast<Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_ann_use_received));

            if(tmp_type_ann_use_received == Common::TYPE_NEURAL_NETWORK_ALL) { return; }

            if(tmp_type_ann_use_received == Common::TYPE_NEURAL_NETWORK_TRAINER) { this->MSE_CHART->Series["MSE_Series_BesT" + (is_tesTtype_received ? "Test" : "Train") + "_"+(is_long_type_received ? "Long" : "Short")+"_"+ann_received.ToString()]->Points->AddXY(x_received, floor(y_received*pow(10, 5))/pow(10, 5)); }
            else if(tmp_type_ann_use_received == Common::TYPE_NEURAL_NETWORK_TRAINED) { this->MSE_CHART->Series["MSE_Series_Trained_" + (is_tesTtype_received ? "Test" : "Train")+"_"+(is_long_type_received ? "Long" : "Short")+"_"+ann_received.ToString()]->Points->AddXY(x_received, floor(y_received*pow(10, 5))/pow(10, 5)); }
        }

        delegate void Chart_OUTPUTReset__PoinTDelegate(const bool is_long_type_received);
        void CFormNN::Chart_OUTPUTReset__Point(const bool is_long_type_received)
        {
            if(this->InvokeRequired)
            {
                Chart_OUTPUTReset__PoinTDelegate^ tmp_invoke_action = gcnew Chart_OUTPUTReset__PoinTDelegate(this, &CFormNN::Chart_OUTPUTReset__Point);
                
                this->Invoke(tmp_invoke_action, is_long_type_received);
                
                delete(tmp_invoke_action);

                return;
            }

            if(is_long_type_received) { this->OUTPUTCHART->Series["OUTPUTSeries_Long"]->Points->Clear(); }
            else { this->OUTPUTCHART->Series["OUTPUTSeries_Short"]->Points->Clear(); }
        }

        delegate void Chart_OUTPUTAdd_PoinTDelegate(const bool is_long_type_received,
                                                                                            const double x_received,
                                                                                            const double y_received);
        void CFormNN::Chart_OUTPUTAdd_Point(const bool is_long_type_received,
                                                                                const double x_received,
                                                                                const double y_received)
        {
            if(this->InvokeRequired)
            {
                Chart_OUTPUTAdd_PoinTDelegate^ tmp_invoke_action = gcnew Chart_OUTPUTAdd_PoinTDelegate(this, &CFormNN::Chart_OUTPUTAdd_Point);
                
                this->Invoke(tmp_invoke_action, is_long_type_received, x_received, y_received);
                
                delete(tmp_invoke_action);

                return;
            }

            if(is_long_type_received) { this->OUTPUTCHART->Series["OUTPUTSeries_Long"]->Points->AddXY(x_received, floor(y_received*pow(10, 5))/pow(10, 5)); }
            else { this->OUTPUTCHART->Series["OUTPUTSeries_Short"]->Points->AddXY(x_received, floor(y_received*pow(10, 5))/pow(10, 5)); }
        }

        delegate void Chart_OUTPUTScale_Delegate(const bool is_y_axis_received,
                                                                                  const double min_received,
                                                                                  const double max_received);
        void CFormNN::Chart_OUTPUTScale(const bool is_y_axis_received,
                                                                        const double min_received,
                                                                        const double max_received)
        {
            if(this->InvokeRequired)
            {
                Chart_OUTPUTScale_Delegate^ tmp_invoke_action = gcnew Chart_OUTPUTScale_Delegate(this, &CFormNN::Chart_OUTPUTScale);
                
                this->Invoke(tmp_invoke_action, is_y_axis_received, min_received, max_received);
                
                delete(tmp_invoke_action);

                return;
            }

            if(is_y_axis_received)
            {
                this->OUTPUTCHART->ChartAreas["OutpuTChartArea"]->AxisY->Minimum = min_received;
                this->OUTPUTCHART->ChartAreas["OutpuTChartArea"]->AxisY->Maximum = max_received;
            }
            else
            {
                this->OUTPUTCHART->ChartAreas["OutpuTChartArea"]->AxisX->Minimum = min_received;
                this->OUTPUTCHART->ChartAreas["OutpuTChartArea"]->AxisX->Maximum = max_received;
            }
        
        }

        delegate void Chart_WEIGHTAdd_PoinTDelegate(const bool is_long_type_received,
                                                                                          const double x_received,
                                                                                          const double y_received);
        void CFormNN::Chart_WEIGHTAdd_Point(const bool is_long_type_received,
                                                                             const double x_received,
                                                                             const double y_received)
        {
            if(this->InvokeRequired)
            {
                Chart_WEIGHTAdd_PoinTDelegate^ tmp_invoke_action = gcnew Chart_WEIGHTAdd_PoinTDelegate(this, &CFormNN::Chart_WEIGHTAdd_Point);
                
                this->Invoke(tmp_invoke_action, is_long_type_received, x_received, y_received);
                
                delete(tmp_invoke_action);

                return;
            }

            if(is_long_type_received) { this->OUTPUTCHART->Series["WEIGHTSeries_Long_1"]->Points->AddXY(x_received, floor(y_received*pow(10, 5))/pow(10, 5)); }
            else { this->OUTPUTCHART->Series["WEIGHTSeries_ShorT1"]->Points->AddXY(x_received, floor(y_received*pow(10, 5))/pow(10, 5)); }
        }

        void CFormNN::ParameterOkButton_Click(System::Object^  sender, System::EventArgs^  e)
        {
            TableLayoutColumnStyleCollection^ tmp_ptr_styles = this->NNFormLayoutPanel->ColumnStyles;

            for(int i(0); i < tmp_ptr_styles->Count; ++i)
            {
                bool tmp_boolean(false);

                ColumnStyle^ tmp_ptr_row_style = tmp_ptr_styles[i];

                switch(i)
                {
                    // Neural Network TabPage
                    case 0: tmp_boolean = true; break;
                    case 1: // Neural Network Parameter
                        this->ParameterPanel->Enabled = false;
                        this->ParameterPanel->Visible = false;
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

            this->_parameter_init = true;
        }

        void CFormNN::ParameterResetButton_Click(System::Object^  sender, System::EventArgs^  e)
        { this->Parameter_Reset(); }

        void CFormNN::Parameter_Reset(void)
        {
            this->NeuralNetworkTypeComboBox->SelectedIndex = 1;
            this->ActivationFunctionHiddenComboBox->SelectedIndex = 19;
            this->ActivationFunctionOutputComboBox->SelectedIndex = 21;
            this->ErrorFunctionComboBox->SelectedIndex = 2;
            this->StopFunctionComboBox->SelectedIndex = 0;
            this->TrainComboBox->SelectedIndex = 2;

            this->NumberInputTextBox->Text = "1";
            this->NumberOutputTextBox->Text = "1";
            this->NumberTimeTextBox->Text = "1";
            this->DesiredErrorTextBox->Text = "10e-5";

            this->UseDenormalizeCheckBox->Checked = false;
            this->UseParallelCheckBox->Checked = false;
            this->UseGPUCheckBox->Checked = false;
            this->UseCascadeTrainingCheckBox->Checked = false;
            this->UseGeneticAlgorithmCheckBox->Checked = false;
        }

        void CFormNN::ParameterSaveButton_Click(System::Object^  sender, System::EventArgs^  e)
        {

        }
        
        delegate void delegate__CFormNN__Exit(const bool exit_thread_received);
        void CFormNN::Exit(const bool exit_thread_received)
        {
            if(this->InvokeRequired) // Thread safe
            {
                delegate__CFormNN__Exit ^tmp_invoke_action = gcnew delegate__CFormNN__Exit(this, &CFormNN::Exit);
                
                this->Invoke(tmp_invoke_action, exit_thread_received);
                
                delete(tmp_invoke_action);

                return;
            }
            
            this->isThreading = false;

            if(exit_thread_received) { ExitThread(0); } else { Close(); }
        }

        CFormNN::~CFormNN()
        {
            if(p_Ptr_Ressource_Manager) { delete(p_Ptr_Ressource_Manager); }
            if(components) { delete(components); }
        }
    }
}