#pragma once

#include <Enums/Enum_Type_Neural_Network_Use.hpp>

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
        using namespace System::Collections::Generic;
        using namespace System::Windows::Forms::DataVisualization::Charting;

        /// <summary>
        /// Summary for CFormNN
        /// </summary>
        public ref class CFormNN : public System::Windows::Forms::Form
        {
        protected:
            System::Resources::ResourceManager^ p_Ptr_Ressource_Manager;

        public:
            CFormNN(System::Void);

            System::Void Exit(const System::Boolean exiTprocess_received);
            System::Void Maximized(System::Void);
            System::Void FormClose(System::Void);
            System::Void Cin_NN_Info(System::String^ cin_texTreceived);
            System::Void Chart_MSE_Init_(const System::Boolean is_long_type_received, System::UInt32 counTseries_received, const System::Double counTinterval_datapoint);
            System::Void Chart_MSE_Interval(const System::Double counTinterval_datapoint);
            System::Void Chart_MSE_Set_Info(const System::UInt16 type_received, System::String^ texTreceived);
            System::Void Chart_MSE_Set_Title_X(System::String^ title_X_received);
            System::Void Chart_MSE_Reset__Point(const System::Boolean is_long_type_received);
            System::Void Chart_MSE_Add_Point(const System::Byte type_ann_use_received,
                                                                        const System::Boolean is_long_type_received,
                                                                        const System::Boolean is_tesTtype_received,
                                                                        const System::UInt32 ann_received,
                                                                        const System::Double x_received,
                                                                        const System::Double y_received);
            System::Void Chart_OUTPUTReset__Point(const System::Boolean is_long_type_received);
            System::Void Chart_OUTPUTAdd_Point(const System::Boolean is_long_type_received,
                                                                                const System::Double x_received,
                                                                                const System::Double y_received);
            System::Void Chart_OUTPUTScale(const System::Boolean is_y_axis_received,
                                                                        const System::Double min_received,
                                                                        const System::Double max_received);
            System::Void Chart_WEIGHTAdd_Point(const System::Boolean is_long_type_received,
                                                                            const System::Double x_received,
                                                                            const System::Double y_received);
            
            const System::Boolean Get__Parameter_Init_(System::Void);
            System::Boolean isThreading;

        private:
            System::Boolean _moving;
            System::Boolean _parameter_init;

            System::UInt32 _counTindex;

            System::Void Parameter_Reset(System::Void);
            private: System::Windows::Forms::TabPage^  WEIGHTTabPage;

            private: System::Windows::Forms::DataVisualization::Charting::Chart^  WEIGHTCHART;
            private: System::Windows::Forms::TableLayoutPanel^  MSE_INFO_TableLayoutPanel;
            private: System::Windows::Forms::Label^  MSE_Current_Epoch_Done_Label;
            private: System::Windows::Forms::Label^  MSE_Total_Epoch_Done_Label;
            private: System::Windows::Forms::Label^  MSE_Current_Label;
            private: System::Windows::Forms::Label^  MSE_Lower_Label;
            private: System::Windows::Forms::Label^  MSE_Higher_Label;
            private: System::Windows::Forms::Label^  MSE_Lower_Low_Label;
            private: System::Windows::Forms::Label^  MSE_Higher_High_Label;
            private: System::Windows::Forms::Label^  MSE_ANNS_TRAINING_Label;
            private: System::Windows::Forms::TabPage^  INFO_TabPage;

            private: System::Windows::Forms::Label^  INFO_LABEL;
            private: System::Windows::Forms::TableLayoutPanel^  ParameterTableLayoutPanel;
            private: System::Windows::Forms::Panel^  ParameterPanel;
            private: System::Windows::Forms::ComboBox^  ActivationFunctionOutputComboBox;
            private: System::Windows::Forms::Label^  ActivationFunctionHiddenLayerLabel;

            private: System::Windows::Forms::Label^  ActivationFunctionOutputLabel;


            private: System::Windows::Forms::ComboBox^  ActivationFunctionHiddenComboBox;
            private: System::Windows::Forms::Button^  ParameterOkButton;
            private: System::Windows::Forms::Button^  ParameterResetButton;


            private: System::Windows::Forms::Button^  ParameterSaveButton;
            private: System::Windows::Forms::Label^  IndicatorLabel;
            private: System::Windows::Forms::Label^  TrainLabel;

            private: System::Windows::Forms::Label^  StopFunctionLabel;

            private: System::Windows::Forms::ComboBox^  TrainComboBox;
            private: System::Windows::Forms::Label^  ErrorFunctionLabel;



            private: System::Windows::Forms::ComboBox^  StopFunctionComboBox;

            private: System::Windows::Forms::ComboBox^  ErrorFunctionComboBox;
            private: System::Windows::Forms::CheckBox^  UseGeneticAlgorithmCheckBox;
            private: System::Windows::Forms::CheckBox^  UseCascadeTrainingCheckBox;
            private: System::Windows::Forms::CheckBox^  UseGPUCheckBox;
            private: System::Windows::Forms::CheckBox^  UseParallelCheckBox;






            private: System::Windows::Forms::CheckBox^  UseDenormalizeCheckBox;
            private: System::Windows::Forms::TextBox^  NumberInputTextBox;
            private: System::Windows::Forms::TextBox^  DesiredErrorTextBox;


            private: System::Windows::Forms::TextBox^  NumberTimeTextBox;

            private: System::Windows::Forms::TextBox^  NumberOutputTextBox;

            private: System::Windows::Forms::Label^  DesiredErrorLabel;

            private: System::Windows::Forms::Label^  NumberTimeLabel;

            private: System::Windows::Forms::Label^  NumberOutputLabel;

            private: System::Windows::Forms::Label^  NumberInputLabel;
            private: System::Windows::Forms::Label^  NeuralNetworkTypeLabel;
            private: System::Windows::Forms::ComboBox^  NeuralNetworkTypeComboBox;

            System::Drawing::Point _position_offset;

        protected:
            /// <summary>
            /// Clean up any resources being used.
            /// </summary>
            ~CFormNN();

            private: System::Windows::Forms::TabControl^  NN_TabControl;
            private: System::Windows::Forms::TabPage^  MSE_TabPage;
            private: System::Windows::Forms::TabPage^  OUTPUTTabPage;

            private: System::Windows::Forms::DataVisualization::Charting::Chart^  OUTPUTCHART;
            private: System::Windows::Forms::TableLayoutPanel^  NNFormLayoutPanel;
            private: System::Windows::Forms::DataVisualization::Charting::Chart^  MSE_CHART;
            private: System::Windows::Forms::TableLayoutPanel^  NNSecondaryTableLayoutPanel;
            private: System::Windows::Forms::TableLayoutPanel^  NNTopTableLayoutPanel;
            private: System::Windows::Forms::Button^  NN_BUTTON_CLOSE;
            private: System::Windows::Forms::Label^  NN_TITLE_LABEL;
            private: System::Windows::Forms::TableLayoutPanel^  NNPrimaryTableLayoutPanel;
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
                System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea1 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
                System::Windows::Forms::DataVisualization::Charting::Title^  title1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
                System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea2 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
                System::Windows::Forms::DataVisualization::Charting::Series^  series1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
                System::Windows::Forms::DataVisualization::Charting::Series^  series2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
                System::Windows::Forms::DataVisualization::Charting::Title^  title2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
                System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea3 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
                System::Windows::Forms::DataVisualization::Charting::Series^  series3 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
                System::Windows::Forms::DataVisualization::Charting::Series^  series4 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
                System::Windows::Forms::DataVisualization::Charting::Series^  series5 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
                System::Windows::Forms::DataVisualization::Charting::Series^  series6 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
                System::Windows::Forms::DataVisualization::Charting::Title^  title3 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
                this->NNSecondaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->NNTopTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->NN_BUTTON_CLOSE = (gcnew System::Windows::Forms::Button());
                this->NN_TITLE_LABEL = (gcnew System::Windows::Forms::Label());
                this->NNFormLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->NN_TabControl = (gcnew System::Windows::Forms::TabControl());
                this->MSE_TabPage = (gcnew System::Windows::Forms::TabPage());
                this->MSE_INFO_TableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->MSE_Current_Epoch_Done_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_Total_Epoch_Done_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_Current_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_Lower_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_Higher_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_Lower_Low_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_Higher_High_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_ANNS_TRAINING_Label = (gcnew System::Windows::Forms::Label());
                this->MSE_CHART = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->OUTPUTTabPage = (gcnew System::Windows::Forms::TabPage());
                this->OUTPUTCHART = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->WEIGHTTabPage = (gcnew System::Windows::Forms::TabPage());
                this->WEIGHTCHART = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->INFO_TabPage = (gcnew System::Windows::Forms::TabPage());
                this->INFO_LABEL = (gcnew System::Windows::Forms::Label());
                this->ParameterTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->ParameterPanel = (gcnew System::Windows::Forms::Panel());
                this->DesiredErrorTextBox = (gcnew System::Windows::Forms::TextBox());
                this->NumberTimeTextBox = (gcnew System::Windows::Forms::TextBox());
                this->NumberOutputTextBox = (gcnew System::Windows::Forms::TextBox());
                this->NumberInputTextBox = (gcnew System::Windows::Forms::TextBox());
                this->UseGeneticAlgorithmCheckBox = (gcnew System::Windows::Forms::CheckBox());
                this->UseCascadeTrainingCheckBox = (gcnew System::Windows::Forms::CheckBox());
                this->UseGPUCheckBox = (gcnew System::Windows::Forms::CheckBox());
                this->UseParallelCheckBox = (gcnew System::Windows::Forms::CheckBox());
                this->UseDenormalizeCheckBox = (gcnew System::Windows::Forms::CheckBox());
                this->IndicatorLabel = (gcnew System::Windows::Forms::Label());
                this->ParameterResetButton = (gcnew System::Windows::Forms::Button());
                this->ParameterSaveButton = (gcnew System::Windows::Forms::Button());
                this->ParameterOkButton = (gcnew System::Windows::Forms::Button());
                this->NeuralNetworkTypeLabel = (gcnew System::Windows::Forms::Label());
                this->ActivationFunctionHiddenLayerLabel = (gcnew System::Windows::Forms::Label());
                this->DesiredErrorLabel = (gcnew System::Windows::Forms::Label());
                this->NumberTimeLabel = (gcnew System::Windows::Forms::Label());
                this->NumberOutputLabel = (gcnew System::Windows::Forms::Label());
                this->NumberInputLabel = (gcnew System::Windows::Forms::Label());
                this->TrainLabel = (gcnew System::Windows::Forms::Label());
                this->StopFunctionLabel = (gcnew System::Windows::Forms::Label());
                this->TrainComboBox = (gcnew System::Windows::Forms::ComboBox());
                this->ErrorFunctionLabel = (gcnew System::Windows::Forms::Label());
                this->StopFunctionComboBox = (gcnew System::Windows::Forms::ComboBox());
                this->ActivationFunctionOutputLabel = (gcnew System::Windows::Forms::Label());
                this->NeuralNetworkTypeComboBox = (gcnew System::Windows::Forms::ComboBox());
                this->ErrorFunctionComboBox = (gcnew System::Windows::Forms::ComboBox());
                this->ActivationFunctionHiddenComboBox = (gcnew System::Windows::Forms::ComboBox());
                this->ActivationFunctionOutputComboBox = (gcnew System::Windows::Forms::ComboBox());
                this->NNPrimaryTableLayoutPanel = (gcnew System::Windows::Forms::TableLayoutPanel());
                this->NNSecondaryTableLayoutPanel->SuspendLayout();
                this->NNTopTableLayoutPanel->SuspendLayout();
                this->NNFormLayoutPanel->SuspendLayout();
                this->NN_TabControl->SuspendLayout();
                this->MSE_TabPage->SuspendLayout();
                this->MSE_INFO_TableLayoutPanel->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->MSE_CHART))->BeginInit();
                this->OUTPUTTabPage->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->OUTPUTCHART))->BeginInit();
                this->WEIGHTTabPage->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->WEIGHTCHART))->BeginInit();
                this->INFO_TabPage->SuspendLayout();
                this->ParameterTableLayoutPanel->SuspendLayout();
                this->ParameterPanel->SuspendLayout();
                this->NNPrimaryTableLayoutPanel->SuspendLayout();
                this->SuspendLayout();
                // 
                // NNSecondaryTableLayoutPanel
                // 
                this->NNSecondaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NNSecondaryTableLayoutPanel->ColumnCount = 1;
                this->NNSecondaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNSecondaryTableLayoutPanel->Controls->Add(this->NNTopTableLayoutPanel, 0, 0);
                this->NNSecondaryTableLayoutPanel->Controls->Add(this->NNFormLayoutPanel, 0, 1);
                this->NNSecondaryTableLayoutPanel->Location = System::Drawing::Point(1, 1);
                this->NNSecondaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->NNSecondaryTableLayoutPanel->Name = L"NNSecondaryTableLayoutPanel";
                this->NNSecondaryTableLayoutPanel->RowCount = 2;
                this->NNSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->NNSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNSecondaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->NNSecondaryTableLayoutPanel->Size = System::Drawing::Size(863, 533);
                this->NNSecondaryTableLayoutPanel->TabIndex = 1;
                // 
                // NNTopTableLayoutPanel
                // 
                this->NNTopTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NNTopTableLayoutPanel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(62)),
                    static_cast<System::Int32>(static_cast<System::Byte>(62)), static_cast<System::Int32>(static_cast<System::Byte>(66)));
                this->NNTopTableLayoutPanel->ColumnCount = 3;
                this->NNTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    125)));
                this->NNTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNTopTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    30)));
                this->NNTopTableLayoutPanel->Controls->Add(this->NN_BUTTON_CLOSE, 2, 0);
                this->NNTopTableLayoutPanel->Controls->Add(this->NN_TITLE_LABEL, 0, 0);
                this->NNTopTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->NNTopTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->NNTopTableLayoutPanel->Name = L"NNTopTableLayoutPanel";
                this->NNTopTableLayoutPanel->RowCount = 1;
                this->NNTopTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNTopTableLayoutPanel->Size = System::Drawing::Size(863, 30);
                this->NNTopTableLayoutPanel->TabIndex = 1;
                this->NNTopTableLayoutPanel->DoubleClick += gcnew System::EventHandler(this, &CFormNN::NNTopTableLayoutPanel_DoubleClick);
                this->NNTopTableLayoutPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormNN::FormNN_Move_MouseDown);
                this->NNTopTableLayoutPanel->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormNN::FormNN_Move_MouseMove);
                this->NNTopTableLayoutPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormNN::FormNN_Move_MouseUp);
                // 
                // NN_BUTTON_CLOSE
                // 
                this->NN_BUTTON_CLOSE->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NN_BUTTON_CLOSE->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
                this->NN_BUTTON_CLOSE->FlatAppearance->BorderSize = 0;
                this->NN_BUTTON_CLOSE->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->NN_BUTTON_CLOSE->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                    static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->NN_BUTTON_CLOSE->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->NN_BUTTON_CLOSE->Location = System::Drawing::Point(833, 0);
                this->NN_BUTTON_CLOSE->Margin = System::Windows::Forms::Padding(0);
                this->NN_BUTTON_CLOSE->Name = L"NN_BUTTON_CLOSE";
                this->NN_BUTTON_CLOSE->Size = System::Drawing::Size(30, 30);
                this->NN_BUTTON_CLOSE->TabIndex = 1;
                this->NN_BUTTON_CLOSE->UseVisualStyleBackColor = false;
                this->NN_BUTTON_CLOSE->Click += gcnew System::EventHandler(this, &CFormNN::NN_BUTTON_CLOSE_Click);
                // 
                // NN_TITLE_LABEL
                // 
                this->NN_TITLE_LABEL->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NN_TITLE_LABEL->Font = (gcnew System::Drawing::Font(L"Arial", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NN_TITLE_LABEL->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(153)), static_cast<System::Int32>(static_cast<System::Byte>(153)),
                    static_cast<System::Int32>(static_cast<System::Byte>(153)));
                this->NN_TITLE_LABEL->Location = System::Drawing::Point(0, 0);
                this->NN_TITLE_LABEL->Margin = System::Windows::Forms::Padding(0);
                this->NN_TITLE_LABEL->Name = L"NN_TITLE_LABEL";
                this->NN_TITLE_LABEL->Size = System::Drawing::Size(125, 30);
                this->NN_TITLE_LABEL->TabIndex = 1;
                this->NN_TITLE_LABEL->Text = L"MyEA - NN";
                this->NN_TITLE_LABEL->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                this->NN_TITLE_LABEL->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormNN::FormNN_Move_MouseDown);
                this->NN_TITLE_LABEL->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormNN::FormNN_Move_MouseMove);
                this->NN_TITLE_LABEL->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &CFormNN::FormNN_Move_MouseUp);
                // 
                // NNFormLayoutPanel
                // 
                this->NNFormLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NNFormLayoutPanel->ColumnCount = 2;
                this->NNFormLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    0)));
                this->NNFormLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNFormLayoutPanel->Controls->Add(this->NN_TabControl, 0, 0);
                this->NNFormLayoutPanel->Controls->Add(this->ParameterTableLayoutPanel, 1, 0);
                this->NNFormLayoutPanel->Location = System::Drawing::Point(0, 30);
                this->NNFormLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->NNFormLayoutPanel->Name = L"NNFormLayoutPanel";
                this->NNFormLayoutPanel->RowCount = 1;
                this->NNFormLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 100)));
                this->NNFormLayoutPanel->Size = System::Drawing::Size(863, 503);
                this->NNFormLayoutPanel->TabIndex = 5;
                // 
                // NN_TabControl
                // 
                this->NN_TabControl->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NN_TabControl->Controls->Add(this->MSE_TabPage);
                this->NN_TabControl->Controls->Add(this->OUTPUTTabPage);
                this->NN_TabControl->Controls->Add(this->WEIGHTTabPage);
                this->NN_TabControl->Controls->Add(this->INFO_TabPage);
                this->NN_TabControl->Location = System::Drawing::Point(0, 0);
                this->NN_TabControl->Margin = System::Windows::Forms::Padding(0);
                this->NN_TabControl->Name = L"NN_TabControl";
                this->NN_TabControl->SelectedIndex = 0;
                this->NN_TabControl->Size = System::Drawing::Size(1, 503);
                this->NN_TabControl->TabIndex = 0;
                // 
                // MSE_TabPage
                // 
                this->MSE_TabPage->Controls->Add(this->MSE_INFO_TableLayoutPanel);
                this->MSE_TabPage->Controls->Add(this->MSE_CHART);
                this->MSE_TabPage->Location = System::Drawing::Point(4, 22);
                this->MSE_TabPage->Name = L"MSE_TabPage";
                this->MSE_TabPage->Padding = System::Windows::Forms::Padding(3);
                this->MSE_TabPage->Size = System::Drawing::Size(0, 477);
                this->MSE_TabPage->TabIndex = 0;
                this->MSE_TabPage->Text = L"MSE";
                this->MSE_TabPage->UseVisualStyleBackColor = true;
                // 
                // MSE_INFO_TableLayoutPanel
                // 
                this->MSE_INFO_TableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_INFO_TableLayoutPanel->ColumnCount = 1;
                this->MSE_INFO_TableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Current_Epoch_Done_Label, 0, 0);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Total_Epoch_Done_Label, 0, 1);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Current_Label, 0, 2);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Lower_Label, 0, 3);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Higher_Label, 0, 4);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Lower_Low_Label, 0, 5);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_Higher_High_Label, 0, 6);
                this->MSE_INFO_TableLayoutPanel->Controls->Add(this->MSE_ANNS_TRAINING_Label, 0, 7);
                this->MSE_INFO_TableLayoutPanel->Location = System::Drawing::Point(-155, 0);
                this->MSE_INFO_TableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->MSE_INFO_TableLayoutPanel->Name = L"MSE_INFO_TableLayoutPanel";
                this->MSE_INFO_TableLayoutPanel->RowCount = 8;
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
                this->MSE_INFO_TableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    20)));
                this->MSE_INFO_TableLayoutPanel->Size = System::Drawing::Size(155, 477);
                this->MSE_INFO_TableLayoutPanel->TabIndex = 1;
                // 
                // MSE_Current_Epoch_Done_Label
                // 
                this->MSE_Current_Epoch_Done_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Current_Epoch_Done_Label->AutoSize = true;
                this->MSE_Current_Epoch_Done_Label->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->MSE_Current_Epoch_Done_Label->Location = System::Drawing::Point(0, 0);
                this->MSE_Current_Epoch_Done_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Current_Epoch_Done_Label->Name = L"MSE_Current_Epoch_Done_Label";
                this->MSE_Current_Epoch_Done_Label->Size = System::Drawing::Size(155, 39);
                this->MSE_Current_Epoch_Done_Label->TabIndex = 0;
                this->MSE_Current_Epoch_Done_Label->Text = L"Epoch\r\n[NaN / NaN]\r\nNaN%";
                // 
                // MSE_Total_Epoch_Done_Label
                // 
                this->MSE_Total_Epoch_Done_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Total_Epoch_Done_Label->AutoSize = true;
                this->MSE_Total_Epoch_Done_Label->Location = System::Drawing::Point(0, 59);
                this->MSE_Total_Epoch_Done_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Total_Epoch_Done_Label->Name = L"MSE_Total_Epoch_Done_Label";
                this->MSE_Total_Epoch_Done_Label->Size = System::Drawing::Size(155, 39);
                this->MSE_Total_Epoch_Done_Label->TabIndex = 1;
                this->MSE_Total_Epoch_Done_Label->Text = L"Epoch Total\r\n[NaN / NaN]\r\nNaN%";
                // 
                // MSE_Current_Label
                // 
                this->MSE_Current_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Current_Label->AutoSize = true;
                this->MSE_Current_Label->Location = System::Drawing::Point(0, 118);
                this->MSE_Current_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Current_Label->Name = L"MSE_Current_Label";
                this->MSE_Current_Label->Size = System::Drawing::Size(155, 26);
                this->MSE_Current_Label->TabIndex = 2;
                this->MSE_Current_Label->Text = L"MSE Current, Average\r\n[NaN]";
                // 
                // MSE_Lower_Label
                // 
                this->MSE_Lower_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Lower_Label->AutoSize = true;
                this->MSE_Lower_Label->Location = System::Drawing::Point(0, 164);
                this->MSE_Lower_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Lower_Label->Name = L"MSE_Lower_Label";
                this->MSE_Lower_Label->Size = System::Drawing::Size(155, 26);
                this->MSE_Lower_Label->TabIndex = 3;
                this->MSE_Lower_Label->Text = L"MSE Lower\r\n[NaN]";
                // 
                // MSE_Higher_Label
                // 
                this->MSE_Higher_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Higher_Label->AutoSize = true;
                this->MSE_Higher_Label->Location = System::Drawing::Point(0, 210);
                this->MSE_Higher_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Higher_Label->Name = L"MSE_Higher_Label";
                this->MSE_Higher_Label->Size = System::Drawing::Size(155, 26);
                this->MSE_Higher_Label->TabIndex = 4;
                this->MSE_Higher_Label->Text = L"MSE Higher\r\n[NaN]";
                // 
                // MSE_Lower_Low_Label
                // 
                this->MSE_Lower_Low_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Lower_Low_Label->AutoSize = true;
                this->MSE_Lower_Low_Label->Location = System::Drawing::Point(0, 256);
                this->MSE_Lower_Low_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Lower_Low_Label->Name = L"MSE_Lower_Low_Label";
                this->MSE_Lower_Low_Label->Size = System::Drawing::Size(155, 26);
                this->MSE_Lower_Low_Label->TabIndex = 5;
                this->MSE_Lower_Low_Label->Text = L"MSE Lower Low\r\n[NaN]";
                // 
                // MSE_Higher_High_Label
                // 
                this->MSE_Higher_High_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_Higher_High_Label->AutoSize = true;
                this->MSE_Higher_High_Label->Location = System::Drawing::Point(0, 302);
                this->MSE_Higher_High_Label->Margin = System::Windows::Forms::Padding(0, 0, 0, 20);
                this->MSE_Higher_High_Label->Name = L"MSE_Higher_High_Label";
                this->MSE_Higher_High_Label->Size = System::Drawing::Size(155, 26);
                this->MSE_Higher_High_Label->TabIndex = 6;
                this->MSE_Higher_High_Label->Text = L"MSE Higher High\r\n[NaN]";
                // 
                // MSE_ANNS_TRAINING_Label
                // 
                this->MSE_ANNS_TRAINING_Label->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->MSE_ANNS_TRAINING_Label->AutoSize = true;
                this->MSE_ANNS_TRAINING_Label->Location = System::Drawing::Point(0, 348);
                this->MSE_ANNS_TRAINING_Label->Margin = System::Windows::Forms::Padding(0);
                this->MSE_ANNS_TRAINING_Label->Name = L"MSE_ANNS_TRAINING_Label";
                this->MSE_ANNS_TRAINING_Label->Size = System::Drawing::Size(155, 129);
                this->MSE_ANNS_TRAINING_Label->TabIndex = 7;
                this->MSE_ANNS_TRAINING_Label->Text = L"Ann\'s in training\r\n[NaN / NaN]";
                // 
                // MSE_CHART
                // 
                this->MSE_CHART->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                chartArea1->AxisX->InterlacedColor = System::Drawing::Color::White;
                chartArea1->AxisX->Interval = 99;
                chartArea1->AxisX->Minimum = 1;
                chartArea1->AxisX->Title = L"ITERATION(s)";
                chartArea1->AxisY->InterlacedColor = System::Drawing::Color::Black;
                chartArea1->AxisY->Interval = 0.05;
                chartArea1->AxisY->Maximum = 1;
                chartArea1->AxisY->Minimum = 0;
                chartArea1->AxisY->ScaleBreakStyle->StartFromZero = System::Windows::Forms::DataVisualization::Charting::StartFromZero::Yes;
                chartArea1->AxisY->TextOrientation = System::Windows::Forms::DataVisualization::Charting::TextOrientation::Stacked;
                chartArea1->AxisY->Title = L"ERROR";
                chartArea1->BackColor = System::Drawing::Color::White;
                chartArea1->BackSecondaryColor = System::Drawing::Color::White;
                chartArea1->Name = L"MSE_ChartArea";
                this->MSE_CHART->ChartAreas->Add(chartArea1);
                this->MSE_CHART->Location = System::Drawing::Point(0, 0);
                this->MSE_CHART->Margin = System::Windows::Forms::Padding(0);
                this->MSE_CHART->Name = L"MSE_CHART";
                this->MSE_CHART->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                this->MSE_CHART->Size = System::Drawing::Size(0, 477);
                this->MSE_CHART->TabIndex = 0;
                title1->Name = L"MSE_Title_Chart";
                title1->Text = L"Mean Square Error (MSE)";
                this->MSE_CHART->Titles->Add(title1);
                // 
                // OUTPUTTabPage
                // 
                this->OUTPUTTabPage->Controls->Add(this->OUTPUTCHART);
                this->OUTPUTTabPage->Location = System::Drawing::Point(4, 22);
                this->OUTPUTTabPage->Name = L"OUTPUTTabPage";
                this->OUTPUTTabPage->Padding = System::Windows::Forms::Padding(3);
                this->OUTPUTTabPage->Size = System::Drawing::Size(0, 477);
                this->OUTPUTTabPage->TabIndex = 1;
                this->OUTPUTTabPage->Text = L"OUTPUT";
                this->OUTPUTTabPage->UseVisualStyleBackColor = true;
                // 
                // OUTPUTCHART
                // 
                this->OUTPUTCHART->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                chartArea2->AxisX->Interval = 1;
                chartArea2->AxisX->Maximum = 100;
                chartArea2->AxisX->Minimum = 0;
                chartArea2->AxisX->Title = L"ITERATION(s)";
                chartArea2->AxisY->InterlacedColor = System::Drawing::Color::Black;
                chartArea2->AxisY->Interval = 0.1;
                chartArea2->AxisY->Maximum = 1;
                chartArea2->AxisY->Minimum = -1;
                chartArea2->AxisY->TextOrientation = System::Windows::Forms::DataVisualization::Charting::TextOrientation::Stacked;
                chartArea2->AxisY->Title = L"OUTPUT";
                chartArea2->Name = L"OutpuTChartArea";
                this->OUTPUTCHART->ChartAreas->Add(chartArea2);
                this->OUTPUTCHART->Location = System::Drawing::Point(0, 0);
                this->OUTPUTCHART->Margin = System::Windows::Forms::Padding(0);
                this->OUTPUTCHART->Name = L"OUTPUTCHART";
                this->OUTPUTCHART->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                series1->ChartArea = L"OutpuTChartArea";
                series1->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
                series1->Color = System::Drawing::Color::Blue;
                series1->Name = L"OUTPUTSeries_Long";
                series2->ChartArea = L"OutpuTChartArea";
                series2->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
                series2->Color = System::Drawing::Color::Red;
                series2->Name = L"OUTPUTSeries_Short";
                this->OUTPUTCHART->Series->Add(series1);
                this->OUTPUTCHART->Series->Add(series2);
                this->OUTPUTCHART->Size = System::Drawing::Size(835, 477);
                this->OUTPUTCHART->TabIndex = 1;
                title2->Name = L"OutpuTTitle_Chart";
                title2->Text = L"Output to (Y * 10 ^ 2)%, Chance To Open Good Position";
                this->OUTPUTCHART->Titles->Add(title2);
                // 
                // WEIGHTTabPage
                // 
                this->WEIGHTTabPage->Controls->Add(this->WEIGHTCHART);
                this->WEIGHTTabPage->Location = System::Drawing::Point(4, 22);
                this->WEIGHTTabPage->Name = L"WEIGHTTabPage";
                this->WEIGHTTabPage->Size = System::Drawing::Size(0, 477);
                this->WEIGHTTabPage->TabIndex = 2;
                this->WEIGHTTabPage->Text = L"WEIGHT";
                this->WEIGHTTabPage->UseVisualStyleBackColor = true;
                // 
                // WEIGHTCHART
                // 
                this->WEIGHTCHART->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                chartArea3->AxisX->Interval = 1;
                chartArea3->AxisX->Minimum = 0;
                chartArea3->AxisX->Title = L"ITERATION(s)";
                chartArea3->AxisY->InterlacedColor = System::Drawing::Color::Black;
                chartArea3->AxisY->Interval = 0.1;
                chartArea3->AxisY->Maximum = 1;
                chartArea3->AxisY->Minimum = -1;
                chartArea3->AxisY->TextOrientation = System::Windows::Forms::DataVisualization::Charting::TextOrientation::Stacked;
                chartArea3->AxisY->Title = L"WEIGHTS";
                chartArea3->Name = L"WeighTChartArea";
                this->WEIGHTCHART->ChartAreas->Add(chartArea3);
                this->WEIGHTCHART->Location = System::Drawing::Point(0, 0);
                this->WEIGHTCHART->Margin = System::Windows::Forms::Padding(0);
                this->WEIGHTCHART->Name = L"WEIGHTCHART";
                this->WEIGHTCHART->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                series3->ChartArea = L"WeighTChartArea";
                series3->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
                series3->Color = System::Drawing::Color::Blue;
                series3->Name = L"WEIGHTSeries_Long_1";
                series4->ChartArea = L"WeighTChartArea";
                series4->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Point;
                series4->Color = System::Drawing::Color::Aqua;
                series4->Name = L"WEIGHTCURRENTSeries_Long_1";
                series5->ChartArea = L"WeighTChartArea";
                series5->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
                series5->Color = System::Drawing::Color::Red;
                series5->Name = L"WEIGHTSeries_ShorT1";
                series6->ChartArea = L"WeighTChartArea";
                series6->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Point;
                series6->Color = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(128)),
                    static_cast<System::Int32>(static_cast<System::Byte>(0)));
                series6->Name = L"WEIGHTCURRENTSeries_ShorT1";
                this->WEIGHTCHART->Series->Add(series3);
                this->WEIGHTCHART->Series->Add(series4);
                this->WEIGHTCHART->Series->Add(series5);
                this->WEIGHTCHART->Series->Add(series6);
                this->WEIGHTCHART->Size = System::Drawing::Size(835, 477);
                this->WEIGHTCHART->TabIndex = 2;
                title3->Name = L"WeighTTitle_Chart";
                title3->Text = L"Weights Value";
                this->WEIGHTCHART->Titles->Add(title3);
                // 
                // INFO_TabPage
                // 
                this->INFO_TabPage->Controls->Add(this->INFO_LABEL);
                this->INFO_TabPage->Location = System::Drawing::Point(4, 22);
                this->INFO_TabPage->Name = L"INFO_TabPage";
                this->INFO_TabPage->Size = System::Drawing::Size(0, 477);
                this->INFO_TabPage->TabIndex = 3;
                this->INFO_TabPage->Text = L"INFO";
                this->INFO_TabPage->UseVisualStyleBackColor = true;
                // 
                // INFO_LABEL
                // 
                this->INFO_LABEL->Location = System::Drawing::Point(0, 0);
                this->INFO_LABEL->Name = L"INFO_LABEL";
                this->INFO_LABEL->Padding = System::Windows::Forms::Padding(5);
                this->INFO_LABEL->Size = System::Drawing::Size(850, 475);
                this->INFO_LABEL->TabIndex = 0;
                this->INFO_LABEL->Text = L"Info";
                this->INFO_LABEL->TextAlign = System::Drawing::ContentAlignment::TopCenter;
                // 
                // ParameterTableLayoutPanel
                // 
                this->ParameterTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->ParameterTableLayoutPanel->ColumnCount = 1;
                this->ParameterTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->ParameterTableLayoutPanel->Controls->Add(this->ParameterPanel, 0, 0);
                this->ParameterTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->ParameterTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->ParameterTableLayoutPanel->Name = L"ParameterTableLayoutPanel";
                this->ParameterTableLayoutPanel->RowCount = 1;
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    503)));
                this->ParameterTableLayoutPanel->Size = System::Drawing::Size(863, 503);
                this->ParameterTableLayoutPanel->TabIndex = 1;
                // 
                // ParameterPanel
                // 
                this->ParameterPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->ParameterPanel->AutoScroll = true;
                this->ParameterPanel->Controls->Add(this->DesiredErrorTextBox);
                this->ParameterPanel->Controls->Add(this->NumberTimeTextBox);
                this->ParameterPanel->Controls->Add(this->NumberOutputTextBox);
                this->ParameterPanel->Controls->Add(this->NumberInputTextBox);
                this->ParameterPanel->Controls->Add(this->UseGeneticAlgorithmCheckBox);
                this->ParameterPanel->Controls->Add(this->UseCascadeTrainingCheckBox);
                this->ParameterPanel->Controls->Add(this->UseGPUCheckBox);
                this->ParameterPanel->Controls->Add(this->UseParallelCheckBox);
                this->ParameterPanel->Controls->Add(this->UseDenormalizeCheckBox);
                this->ParameterPanel->Controls->Add(this->IndicatorLabel);
                this->ParameterPanel->Controls->Add(this->ParameterResetButton);
                this->ParameterPanel->Controls->Add(this->ParameterSaveButton);
                this->ParameterPanel->Controls->Add(this->ParameterOkButton);
                this->ParameterPanel->Controls->Add(this->NeuralNetworkTypeLabel);
                this->ParameterPanel->Controls->Add(this->ActivationFunctionHiddenLayerLabel);
                this->ParameterPanel->Controls->Add(this->DesiredErrorLabel);
                this->ParameterPanel->Controls->Add(this->NumberTimeLabel);
                this->ParameterPanel->Controls->Add(this->NumberOutputLabel);
                this->ParameterPanel->Controls->Add(this->NumberInputLabel);
                this->ParameterPanel->Controls->Add(this->TrainLabel);
                this->ParameterPanel->Controls->Add(this->StopFunctionLabel);
                this->ParameterPanel->Controls->Add(this->TrainComboBox);
                this->ParameterPanel->Controls->Add(this->ErrorFunctionLabel);
                this->ParameterPanel->Controls->Add(this->StopFunctionComboBox);
                this->ParameterPanel->Controls->Add(this->ActivationFunctionOutputLabel);
                this->ParameterPanel->Controls->Add(this->NeuralNetworkTypeComboBox);
                this->ParameterPanel->Controls->Add(this->ErrorFunctionComboBox);
                this->ParameterPanel->Controls->Add(this->ActivationFunctionHiddenComboBox);
                this->ParameterPanel->Controls->Add(this->ActivationFunctionOutputComboBox);
                this->ParameterPanel->Location = System::Drawing::Point(0, 0);
                this->ParameterPanel->Margin = System::Windows::Forms::Padding(0);
                this->ParameterPanel->Name = L"ParameterPanel";
                this->ParameterPanel->Size = System::Drawing::Size(863, 503);
                this->ParameterPanel->TabIndex = 0;
                // 
                // DesiredErrorTextBox
                // 
                this->DesiredErrorTextBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->DesiredErrorTextBox->Location = System::Drawing::Point(202, 330);
                this->DesiredErrorTextBox->MaxLength = 255;
                this->DesiredErrorTextBox->Name = L"DesiredErrorTextBox";
                this->DesiredErrorTextBox->Size = System::Drawing::Size(232, 20);
                this->DesiredErrorTextBox->TabIndex = 5;
                // 
                // NumberTimeTextBox
                // 
                this->NumberTimeTextBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NumberTimeTextBox->Location = System::Drawing::Point(202, 300);
                this->NumberTimeTextBox->MaxLength = 255;
                this->NumberTimeTextBox->Name = L"NumberTimeTextBox";
                this->NumberTimeTextBox->Size = System::Drawing::Size(232, 20);
                this->NumberTimeTextBox->TabIndex = 5;
                // 
                // NumberOutputTextBox
                // 
                this->NumberOutputTextBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NumberOutputTextBox->Location = System::Drawing::Point(202, 270);
                this->NumberOutputTextBox->MaxLength = 255;
                this->NumberOutputTextBox->Name = L"NumberOutputTextBox";
                this->NumberOutputTextBox->Size = System::Drawing::Size(232, 20);
                this->NumberOutputTextBox->TabIndex = 5;
                // 
                // NumberInputTextBox
                // 
                this->NumberInputTextBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NumberInputTextBox->Location = System::Drawing::Point(202, 240);
                this->NumberInputTextBox->MaxLength = 255;
                this->NumberInputTextBox->Name = L"NumberInputTextBox";
                this->NumberInputTextBox->Size = System::Drawing::Size(232, 20);
                this->NumberInputTextBox->TabIndex = 5;
                // 
                // UseGeneticAlgorithmCheckBox
                // 
                this->UseGeneticAlgorithmCheckBox->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->UseGeneticAlgorithmCheckBox->ForeColor = System::Drawing::Color::White;
                this->UseGeneticAlgorithmCheckBox->Location = System::Drawing::Point(476, 181);
                this->UseGeneticAlgorithmCheckBox->Name = L"UseGeneticAlgorithmCheckBox";
                this->UseGeneticAlgorithmCheckBox->Size = System::Drawing::Size(200, 21);
                this->UseGeneticAlgorithmCheckBox->TabIndex = 4;
                this->UseGeneticAlgorithmCheckBox->Text = L"Use genetic algorithm";
                this->UseGeneticAlgorithmCheckBox->UseVisualStyleBackColor = true;
                // 
                // UseCascadeTrainingCheckBox
                // 
                this->UseCascadeTrainingCheckBox->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->UseCascadeTrainingCheckBox->ForeColor = System::Drawing::Color::White;
                this->UseCascadeTrainingCheckBox->Location = System::Drawing::Point(476, 151);
                this->UseCascadeTrainingCheckBox->Name = L"UseCascadeTrainingCheckBox";
                this->UseCascadeTrainingCheckBox->Size = System::Drawing::Size(200, 21);
                this->UseCascadeTrainingCheckBox->TabIndex = 4;
                this->UseCascadeTrainingCheckBox->Text = L"Use cascade training";
                this->UseCascadeTrainingCheckBox->UseVisualStyleBackColor = true;
                // 
                // UseGPUCheckBox
                // 
                this->UseGPUCheckBox->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->UseGPUCheckBox->ForeColor = System::Drawing::Color::White;
                this->UseGPUCheckBox->Location = System::Drawing::Point(476, 121);
                this->UseGPUCheckBox->Name = L"UseGPUCheckBox";
                this->UseGPUCheckBox->Size = System::Drawing::Size(200, 21);
                this->UseGPUCheckBox->TabIndex = 4;
                this->UseGPUCheckBox->Text = L"Use GPU";
                this->UseGPUCheckBox->UseVisualStyleBackColor = true;
                // 
                // UseParallelCheckBox
                // 
                this->UseParallelCheckBox->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->UseParallelCheckBox->ForeColor = System::Drawing::Color::White;
                this->UseParallelCheckBox->Location = System::Drawing::Point(476, 91);
                this->UseParallelCheckBox->Name = L"UseParallelCheckBox";
                this->UseParallelCheckBox->Size = System::Drawing::Size(200, 21);
                this->UseParallelCheckBox->TabIndex = 4;
                this->UseParallelCheckBox->Text = L"Use parallel";
                this->UseParallelCheckBox->UseVisualStyleBackColor = true;
                // 
                // UseDenormalizeCheckBox
                // 
                this->UseDenormalizeCheckBox->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->UseDenormalizeCheckBox->ForeColor = System::Drawing::Color::White;
                this->UseDenormalizeCheckBox->Location = System::Drawing::Point(476, 60);
                this->UseDenormalizeCheckBox->Name = L"UseDenormalizeCheckBox";
                this->UseDenormalizeCheckBox->Size = System::Drawing::Size(200, 21);
                this->UseDenormalizeCheckBox->TabIndex = 4;
                this->UseDenormalizeCheckBox->Text = L"Use denormalize";
                this->UseDenormalizeCheckBox->UseVisualStyleBackColor = true;
                // 
                // IndicatorLabel
                // 
                this->IndicatorLabel->AutoSize = true;
                this->IndicatorLabel->Font = (gcnew System::Drawing::Font(L"Arial", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->IndicatorLabel->ForeColor = System::Drawing::Color::White;
                this->IndicatorLabel->Location = System::Drawing::Point(3, 13);
                this->IndicatorLabel->Name = L"IndicatorLabel";
                this->IndicatorLabel->Size = System::Drawing::Size(138, 22);
                this->IndicatorLabel->TabIndex = 3;
                this->IndicatorLabel->Text = L"Indicator Name";
                this->IndicatorLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // ParameterResetButton
                // 
                this->ParameterResetButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->ParameterResetButton->FlatAppearance->BorderColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ParameterResetButton->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ParameterResetButton->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                    static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->ParameterResetButton->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->ParameterResetButton->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ParameterResetButton->ForeColor = System::Drawing::Color::White;
                this->ParameterResetButton->Location = System::Drawing::Point(727, 15);
                this->ParameterResetButton->Name = L"ParameterResetButton";
                this->ParameterResetButton->Size = System::Drawing::Size(125, 23);
                this->ParameterResetButton->TabIndex = 2;
                this->ParameterResetButton->Text = L"Reset";
                this->ParameterResetButton->UseVisualStyleBackColor = true;
                this->ParameterResetButton->Click += gcnew System::EventHandler(this, &CFormNN::ParameterResetButton_Click);
                // 
                // ParameterSaveButton
                // 
                this->ParameterSaveButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
                this->ParameterSaveButton->FlatAppearance->BorderColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ParameterSaveButton->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ParameterSaveButton->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                    static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->ParameterSaveButton->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->ParameterSaveButton->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ParameterSaveButton->ForeColor = System::Drawing::Color::White;
                this->ParameterSaveButton->Location = System::Drawing::Point(596, 469);
                this->ParameterSaveButton->Name = L"ParameterSaveButton";
                this->ParameterSaveButton->Size = System::Drawing::Size(125, 23);
                this->ParameterSaveButton->TabIndex = 2;
                this->ParameterSaveButton->Text = L"Save";
                this->ParameterSaveButton->UseVisualStyleBackColor = true;
                this->ParameterSaveButton->Click += gcnew System::EventHandler(this, &CFormNN::ParameterSaveButton_Click);
                // 
                // ParameterOkButton
                // 
                this->ParameterOkButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
                this->ParameterOkButton->FlatAppearance->BorderColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ParameterOkButton->FlatAppearance->MouseDownBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
                    static_cast<System::Int32>(static_cast<System::Byte>(122)), static_cast<System::Int32>(static_cast<System::Byte>(204)));
                this->ParameterOkButton->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(60)),
                    static_cast<System::Int32>(static_cast<System::Byte>(60)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
                this->ParameterOkButton->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
                this->ParameterOkButton->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ParameterOkButton->ForeColor = System::Drawing::Color::White;
                this->ParameterOkButton->Location = System::Drawing::Point(727, 469);
                this->ParameterOkButton->Name = L"ParameterOkButton";
                this->ParameterOkButton->Size = System::Drawing::Size(125, 23);
                this->ParameterOkButton->TabIndex = 2;
                this->ParameterOkButton->Text = L"Ok";
                this->ParameterOkButton->UseVisualStyleBackColor = true;
                this->ParameterOkButton->Click += gcnew System::EventHandler(this, &CFormNN::ParameterOkButton_Click);
                // 
                // NeuralNetworkTypeLabel
                // 
                this->NeuralNetworkTypeLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NeuralNetworkTypeLabel->ForeColor = System::Drawing::Color::White;
                this->NeuralNetworkTypeLabel->Location = System::Drawing::Point(7, 60);
                this->NeuralNetworkTypeLabel->Name = L"NeuralNetworkTypeLabel";
                this->NeuralNetworkTypeLabel->Size = System::Drawing::Size(190, 21);
                this->NeuralNetworkTypeLabel->TabIndex = 1;
                this->NeuralNetworkTypeLabel->Text = L"Neural Network Type :";
                this->NeuralNetworkTypeLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // ActivationFunctionHiddenLayerLabel
                // 
                this->ActivationFunctionHiddenLayerLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ActivationFunctionHiddenLayerLabel->ForeColor = System::Drawing::Color::White;
                this->ActivationFunctionHiddenLayerLabel->Location = System::Drawing::Point(6, 90);
                this->ActivationFunctionHiddenLayerLabel->Name = L"ActivationFunctionHiddenLayerLabel";
                this->ActivationFunctionHiddenLayerLabel->Size = System::Drawing::Size(190, 21);
                this->ActivationFunctionHiddenLayerLabel->TabIndex = 1;
                this->ActivationFunctionHiddenLayerLabel->Text = L"Activation Function Hidden Layer :";
                this->ActivationFunctionHiddenLayerLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // DesiredErrorLabel
                // 
                this->DesiredErrorLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->DesiredErrorLabel->ForeColor = System::Drawing::Color::White;
                this->DesiredErrorLabel->Location = System::Drawing::Point(6, 330);
                this->DesiredErrorLabel->Name = L"DesiredErrorLabel";
                this->DesiredErrorLabel->Size = System::Drawing::Size(190, 21);
                this->DesiredErrorLabel->TabIndex = 1;
                this->DesiredErrorLabel->Text = L"Desired error :";
                this->DesiredErrorLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // NumberTimeLabel
                // 
                this->NumberTimeLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NumberTimeLabel->ForeColor = System::Drawing::Color::White;
                this->NumberTimeLabel->Location = System::Drawing::Point(6, 300);
                this->NumberTimeLabel->Name = L"NumberTimeLabel";
                this->NumberTimeLabel->Size = System::Drawing::Size(190, 21);
                this->NumberTimeLabel->TabIndex = 1;
                this->NumberTimeLabel->Text = L"Number time :";
                this->NumberTimeLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // NumberOutputLabel
                // 
                this->NumberOutputLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NumberOutputLabel->ForeColor = System::Drawing::Color::White;
                this->NumberOutputLabel->Location = System::Drawing::Point(6, 270);
                this->NumberOutputLabel->Name = L"NumberOutputLabel";
                this->NumberOutputLabel->Size = System::Drawing::Size(190, 21);
                this->NumberOutputLabel->TabIndex = 1;
                this->NumberOutputLabel->Text = L"Number Output :";
                this->NumberOutputLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // NumberInputLabel
                // 
                this->NumberInputLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NumberInputLabel->ForeColor = System::Drawing::Color::White;
                this->NumberInputLabel->Location = System::Drawing::Point(6, 240);
                this->NumberInputLabel->Name = L"NumberInputLabel";
                this->NumberInputLabel->Size = System::Drawing::Size(190, 21);
                this->NumberInputLabel->TabIndex = 1;
                this->NumberInputLabel->Text = L"Number Input :";
                this->NumberInputLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // TrainLabel
                // 
                this->TrainLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->TrainLabel->ForeColor = System::Drawing::Color::White;
                this->TrainLabel->Location = System::Drawing::Point(6, 210);
                this->TrainLabel->Name = L"TrainLabel";
                this->TrainLabel->Size = System::Drawing::Size(190, 21);
                this->TrainLabel->TabIndex = 1;
                this->TrainLabel->Text = L"Train :";
                this->TrainLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // StopFunctionLabel
                // 
                this->StopFunctionLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->StopFunctionLabel->ForeColor = System::Drawing::Color::White;
                this->StopFunctionLabel->Location = System::Drawing::Point(6, 180);
                this->StopFunctionLabel->Name = L"StopFunctionLabel";
                this->StopFunctionLabel->Size = System::Drawing::Size(190, 21);
                this->StopFunctionLabel->TabIndex = 1;
                this->StopFunctionLabel->Text = L"Stop Function :";
                this->StopFunctionLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // TrainComboBox
                // 
                this->TrainComboBox->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
                this->TrainComboBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->TrainComboBox->FormattingEnabled = true;
                this->TrainComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(5) {
                    L"BP INCREMENTAL", L"BP BATCH", L"BP iRPROP",
                        L"BP QUICKPROP", L"BPTT iRPROP"
                });
                this->TrainComboBox->Location = System::Drawing::Point(202, 210);
                this->TrainComboBox->Name = L"TrainComboBox";
                this->TrainComboBox->Size = System::Drawing::Size(232, 22);
                this->TrainComboBox->TabIndex = 0;
                // 
                // ErrorFunctionLabel
                // 
                this->ErrorFunctionLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ErrorFunctionLabel->ForeColor = System::Drawing::Color::White;
                this->ErrorFunctionLabel->Location = System::Drawing::Point(6, 150);
                this->ErrorFunctionLabel->Name = L"ErrorFunctionLabel";
                this->ErrorFunctionLabel->Size = System::Drawing::Size(190, 21);
                this->ErrorFunctionLabel->TabIndex = 1;
                this->ErrorFunctionLabel->Text = L"Error Function :";
                this->ErrorFunctionLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // StopFunctionComboBox
                // 
                this->StopFunctionComboBox->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
                this->StopFunctionComboBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->StopFunctionComboBox->FormattingEnabled = true;
                this->StopFunctionComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"MSE", L"BIT" });
                this->StopFunctionComboBox->Location = System::Drawing::Point(202, 180);
                this->StopFunctionComboBox->Name = L"StopFunctionComboBox";
                this->StopFunctionComboBox->Size = System::Drawing::Size(232, 22);
                this->StopFunctionComboBox->TabIndex = 0;
                // 
                // ActivationFunctionOutputLabel
                // 
                this->ActivationFunctionOutputLabel->Font = (gcnew System::Drawing::Font(L"Arial", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ActivationFunctionOutputLabel->ForeColor = System::Drawing::Color::White;
                this->ActivationFunctionOutputLabel->Location = System::Drawing::Point(6, 120);
                this->ActivationFunctionOutputLabel->Name = L"ActivationFunctionOutputLabel";
                this->ActivationFunctionOutputLabel->Size = System::Drawing::Size(190, 21);
                this->ActivationFunctionOutputLabel->TabIndex = 1;
                this->ActivationFunctionOutputLabel->Text = L"Activation Function Output Layer :";
                this->ActivationFunctionOutputLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
                // 
                // NeuralNetworkTypeComboBox
                // 
                this->NeuralNetworkTypeComboBox->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
                this->NeuralNetworkTypeComboBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->NeuralNetworkTypeComboBox->FormattingEnabled = true;
                this->NeuralNetworkTypeComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(7) {
                    L"NONE", L"FEEDFORWARD", L"FEEDFORWARD SHORTCUT",
                        L"UNROLLED RECURRENT", L"HOPFIELD RECURRENT ", L"JORDAN RECURRENT", L"ELMAN RECURRENT"
                });
                this->NeuralNetworkTypeComboBox->Location = System::Drawing::Point(203, 60);
                this->NeuralNetworkTypeComboBox->Name = L"NeuralNetworkTypeComboBox";
                this->NeuralNetworkTypeComboBox->Size = System::Drawing::Size(232, 22);
                this->NeuralNetworkTypeComboBox->TabIndex = 0;
                this->NeuralNetworkTypeComboBox->Tag = L"";
                // 
                // ErrorFunctionComboBox
                // 
                this->ErrorFunctionComboBox->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
                this->ErrorFunctionComboBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                    static_cast<System::Byte>(0)));
                this->ErrorFunctionComboBox->FormattingEnabled = true;
                this->ErrorFunctionComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"LINEAR", L"TANH", L"CROSS ENTROPY" });
                this->ErrorFunctionComboBox->Location = System::Drawing::Point(202, 150);
                this->ErrorFunctionComboBox->Name = L"ErrorFunctionComboBox";
                this->ErrorFunctionComboBox->Size = System::Drawing::Size(232, 22);
                this->ErrorFunctionComboBox->TabIndex = 0;
                // 
                // ActivationFunctionHiddenComboBox
                // 
                this->ActivationFunctionHiddenComboBox->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
                this->ActivationFunctionHiddenComboBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->ActivationFunctionHiddenComboBox->FormattingEnabled = true;
                this->ActivationFunctionHiddenComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(24) {
                    L"LINEAR", L"THRESHOLD",
                        L"THRESHOLD SYMMETRIC", L"SIGMOID", L"SIGMOID STEPWISE", L"SIGMOID SYMMETRIC", L"SIGMOID SYMMETRIC STEPWISE", L"GAUSSIAN", L"GAUSSIAN SYMMETRIC",
                        L"GAUSSIAN STEPWISE", L"ELLIOT", L"ELLIOT_SYMMETRIC", L"LINEAR PIECE", L"LINEAR PIECE SYMMETRIC", L"SIN", L"SIN SYMMETRIC", L"COS",
                        L"COS SYMMETRIC", L"RELU", L"LEAKY RELU", L"PARAMETRIC RELU", L"LSTM", L"GRU", L"SOFTMAX"
                });
                this->ActivationFunctionHiddenComboBox->Location = System::Drawing::Point(202, 90);
                this->ActivationFunctionHiddenComboBox->Name = L"ActivationFunctionHiddenComboBox";
                this->ActivationFunctionHiddenComboBox->Size = System::Drawing::Size(232, 22);
                this->ActivationFunctionHiddenComboBox->TabIndex = 0;
                this->ActivationFunctionHiddenComboBox->Tag = L"";
                // 
                // ActivationFunctionOutputComboBox
                // 
                this->ActivationFunctionOutputComboBox->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
                this->ActivationFunctionOutputComboBox->Font = (gcnew System::Drawing::Font(L"Arial", 8.25F, System::Drawing::FontStyle::Regular,
                    System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->ActivationFunctionOutputComboBox->FormattingEnabled = true;
                this->ActivationFunctionOutputComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(22) {
                    L"LINEAR", L"THRESHOLD",
                        L"THRESHOLD SYMMETRIC", L"SIGMOID", L"SIGMOID STEPWISE", L"SIGMOID SYMMETRIC", L"SIGMOID SYMMETRIC STEPWISE", L"GAUSSIAN", L"GAUSSIAN SYMMETRIC",
                        L"GAUSSIAN STEPWISE", L"ELLIOT", L"ELLIOT_SYMMETRIC", L"LINEAR PIECE", L"LINEAR PIECE SYMMETRIC", L"SIN", L"SIN SYMMETRIC", L"COS",
                        L"COS SYMMETRIC", L"RELU", L"LEAKY RELU", L"PARAMETRIC RELU", L"SOFTMAX"
                });
                this->ActivationFunctionOutputComboBox->Location = System::Drawing::Point(202, 120);
                this->ActivationFunctionOutputComboBox->Name = L"ActivationFunctionOutputComboBox";
                this->ActivationFunctionOutputComboBox->Size = System::Drawing::Size(232, 22);
                this->ActivationFunctionOutputComboBox->TabIndex = 0;
                // 
                // NNPrimaryTableLayoutPanel
                // 
                this->NNPrimaryTableLayoutPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                    | System::Windows::Forms::AnchorStyles::Left)
                    | System::Windows::Forms::AnchorStyles::Right));
                this->NNPrimaryTableLayoutPanel->ColumnCount = 3;
                this->NNPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->NNPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNPrimaryTableLayoutPanel->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->NNPrimaryTableLayoutPanel->Controls->Add(this->NNSecondaryTableLayoutPanel, 1, 1);
                this->NNPrimaryTableLayoutPanel->Location = System::Drawing::Point(0, 0);
                this->NNPrimaryTableLayoutPanel->Margin = System::Windows::Forms::Padding(0);
                this->NNPrimaryTableLayoutPanel->Name = L"NNPrimaryTableLayoutPanel";
                this->NNPrimaryTableLayoutPanel->RowCount = 3;
                this->NNPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->NNPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
                    100)));
                this->NNPrimaryTableLayoutPanel->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute,
                    1)));
                this->NNPrimaryTableLayoutPanel->Size = System::Drawing::Size(865, 535);
                this->NNPrimaryTableLayoutPanel->TabIndex = 2;
                // 
                // CFormNN
                // 
                this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
                this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
                this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
                    static_cast<System::Int32>(static_cast<System::Byte>(48)));
                this->ClientSize = System::Drawing::Size(865, 535);
                this->Controls->Add(this->NNPrimaryTableLayoutPanel);
                this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::None;
                this->MaximizeBox = false;
                this->MinimumSize = System::Drawing::Size(865, 535);
                this->Name = L"CFormNN";
                this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
                this->Text = L"MyEA - NN";
                this->Load += gcnew System::EventHandler(this, &CFormNN::FormNN_Load);
                this->NNSecondaryTableLayoutPanel->ResumeLayout(false);
                this->NNTopTableLayoutPanel->ResumeLayout(false);
                this->NNFormLayoutPanel->ResumeLayout(false);
                this->NN_TabControl->ResumeLayout(false);
                this->MSE_TabPage->ResumeLayout(false);
                this->MSE_INFO_TableLayoutPanel->ResumeLayout(false);
                this->MSE_INFO_TableLayoutPanel->PerformLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->MSE_CHART))->EndInit();
                this->OUTPUTTabPage->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->OUTPUTCHART))->EndInit();
                this->WEIGHTTabPage->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->WEIGHTCHART))->EndInit();
                this->INFO_TabPage->ResumeLayout(false);
                this->ParameterTableLayoutPanel->ResumeLayout(false);
                this->ParameterPanel->ResumeLayout(false);
                this->ParameterPanel->PerformLayout();
                this->NNPrimaryTableLayoutPanel->ResumeLayout(false);
                this->ResumeLayout(false);

            }
    #pragma endregion
        private: System::Void FormNN_Load(System::Object^  sender, System::EventArgs^  e);
        private: System::Void FormNN_Move_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void FormNN_Move_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void FormNN_Move_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
        private: System::Void NN_BUTTON_CLOSE_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void NNTopTableLayoutPanel_DoubleClick(System::Object^  sender, System::EventArgs^  e);
        private: System::Void ParameterOkButton_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void ParameterResetButton_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void ParameterSaveButton_Click(System::Object^  sender, System::EventArgs^  e);
};
    }
}
