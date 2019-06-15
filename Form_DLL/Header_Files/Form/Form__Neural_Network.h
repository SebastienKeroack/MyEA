#pragma once

#include <Enums/Enum_Type_Dataset.hpp>

namespace MyEA
{
    namespace Form
    {
        using namespace System;
        using namespace System::Threading;
        using namespace System::ComponentModel;
        using namespace System::Collections;
        using namespace System::Windows::Forms;
        using namespace System::Data;
        using namespace System::Drawing;
        
        /// <summary>
        /// Summary for Form_Neural_Network
        /// </summary>
        public ref class Form_Neural_Network : public System::Windows::Forms::Form
        {
        public:
            ref struct struct_Mean
            {
                void Set__Total_Means(unsigned int const maximum_means_received)
                {
                    this->total_means = maximum_means_received;

                    this->initialize_bias = false;

                    this->momentum_average = 1.0 / static_cast<double>(maximum_means_received);
                }

                void Add(double const value_received)
                {
                    if(this->initialize_bias == false)
                    {
                        this->initialize_bias = true;

                        this->previous_mean_value = this->current_mean_value = value_received;
                    }

                    this->current_mean_value += this->momentum_average * (value_received - this->current_mean_value);

                    if(++this->number_means % this->total_means == 0u)
                    {
                        this->previous_mean_value = this->current_mean_value;

                        this->number_means = 0u;
                    }
                }

                bool initialize_bias = false;

                unsigned int total_means = 10u;
                unsigned int number_means = 0u;

                double Get__Current_Mean(void) { return(this->current_mean_value); }
                double Get__Past_Mean_Diff(void) { return(this->Get__Current_Mean() - this->previous_mean_value); }
                double current_mean_value = 0.0;
                double previous_mean_value = 0.0;
                double momentum_average = 1.0;
            };

            ref struct struct_Chart_Parameters
            {
                unsigned int number_series = 0u;

                double total_data_point = 0.0;
                double precision = 0.0;
                double block_view_X = 0.0;
                System::Collections::Generic::List<double> y_current_value;
                System::Collections::Generic::List<double> y_maximum_value;
                System::Collections::Generic::List<double> y_minimum_value;

                struct_Mean struct_training_mean;
                struct_Mean struct_validating_mean;
                struct_Mean struct_testing_mean;
            };
        
            Form_Neural_Network(void)
            {
                InitializeComponent();
                //
                //TODO: Add the constructor code here
                //

                this->Chart_Loss->MouseWheel += gcnew MouseEventHandler(this, &Form_Neural_Network::Chart_Loss_MouseWheel);
                this->Chart_Loss->ChartAreas[0u]->AxisX->Minimum = 0.0;

                this->Chart_Accuracy->MouseWheel += gcnew MouseEventHandler(this, &Form_Neural_Network::Chart_Accuracy_MouseWheel);
                this->Chart_Accuracy->ChartAreas[0u]->AxisX->Minimum = 0.0;
                
                this->Chart_Output->MouseWheel += gcnew MouseEventHandler(this, &Form_Neural_Network::Chart_Output_MouseWheel);
                this->Chart_Output->ChartAreas[0u]->AxisX->Minimum = 0.0;
                
                this->Chart_Grid_Search->MouseWheel += gcnew MouseEventHandler(this, &Form_Neural_Network::Chart_Grid_Search_MouseWheel);
                this->Chart_Grid_Search->ChartAreas[0u]->AxisX->Minimum = 0.0;

                this->_signal_training_stop = false;
                this->_signal_training_menu = false;

                this->_ptr_chart_parameters_loss = gcnew ref struct struct_Chart_Parameters;
                this->_ptr_chart_parameters_loss->number_series = 0u;
                this->_ptr_chart_parameters_loss->total_data_point = 0.0;
                this->_ptr_chart_parameters_loss->precision = 0.005;
                this->Label_Loss_Mean->Text = "Mean " + this->_ptr_chart_parameters_loss->struct_training_mean.total_means.ToString();
                
                this->_ptr_chart_parameters_accuracy = gcnew ref struct struct_Chart_Parameters;
                this->_ptr_chart_parameters_accuracy->number_series = 0u;
                this->_ptr_chart_parameters_accuracy->total_data_point = 0.0;
                this->_ptr_chart_parameters_accuracy->precision = 0.005;
                this->Label_Accuracy_Mean->Text = "Mean " + this->_ptr_chart_parameters_accuracy->struct_training_mean.total_means.ToString();

                this->_ptr_chart_parameters_output = gcnew ref struct struct_Chart_Parameters;
                this->_ptr_chart_parameters_output->number_series = 0u;
                this->_ptr_chart_parameters_output->total_data_point = 0.0;

                this->_ptr_chart_parameters_grid_search = gcnew ref struct struct_Chart_Parameters;
                this->_ptr_chart_parameters_grid_search->total_data_point = 0.0;
            }

            System::Boolean Get__Signal_Training_Stop(void) { return(this->_signal_training_stop); }
            System::Boolean Get__Signal_Training_Menu(void) { return(this->_signal_training_menu); }

            System::Void Reset__Signal_Training_Menu(void);
            System::Void Chart_Use_Datapoint_Training(bool const use_datapoint_training_received);
            System::Void Chart_Use_Datapoint_Difference(bool const use_datapoint_difference_received);
            System::Void Chart_Initialize(unsigned int const type_chart_received, unsigned int const number_series_received);
            System::Void Chart_Loss_Initialize(unsigned int const number_series_received);
            System::Void Chart_Accuracy_Initialize(unsigned int const number_series_received);
            System::Void Chart_Output_Initialize(unsigned int const number_series_received);
            System::Void Chart_Grid_Search_Initialize(void);
            System::Void Chart_Total_Means(unsigned int const total_means_received);
            System::Void Chart_Reset(unsigned int const type_chart_received);
            System::Void Chart_Rescale(unsigned int const type_chart_received);
            System::Void Chart_Add_Point(unsigned int const type_chart_received,
                                                          unsigned int const index_series_received,
                                                          unsigned int const type_received,
                                                          double const x_received,
                                                          double const y_received);
            System::Void Chart_Loss_Add_Point(unsigned int const index_series_received,
                                                                   unsigned int const type_loss_received,
                                                                   double const x_received,
                                                                   double y_received);
            System::Void Chart_Loss_Diff(unsigned int const index_series_received,
                                                        unsigned int const type_received,
                                                        double const x_received);
            System::Void Chart_Accuracy_Add_Point(unsigned int const index_series_received,
                                                                         unsigned int const type_accuracy_received,
                                                                         double const x_received,
                                                                         double y_received);
            System::Void Chart_Output_Add_Point(unsigned int const index_series_received,
                                                                      unsigned int const type_accuracy_received,
                                                                      double const x_received,
                                                                      double y_received);
            System::Void Chart_Grid_Search_Add_Point(unsigned int const type_loss_received,
                                                                              double const x_received,
                                                                              double y_received);
            System::Void Chart_Grid_Search_Add_Column(System::String ^value_received);
            System::Void Chart_Grid_Search_Add_Row(unsigned int const index_received, System::String ^value_received);
        public:

        private:
            bool _use_Datapoint_Training = true;
            bool _use_Datapoint_Difference = false;
            bool _signal_training_stop = false;
            bool _signal_training_menu = false;
        private: System::Windows::Forms::Button^  Button_Training_Menu;
        private: System::Windows::Forms::Panel^  Panel_Information;
        private: System::Windows::Forms::Button^  Button_Panel_Information;
        private: System::Windows::Forms::Label^  Label_Loss;



        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Training_Current;

        private: System::Windows::Forms::RichTextBox^  RichTextBox_Output;






        private: System::Windows::Forms::RichTextBox^  RichTextBox_DataPoint_Accuracy;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_DataPoint_Loss;
        private: System::Windows::Forms::Label^  label6;
        private: System::Windows::Forms::Label^  Label_Loss_Mean;

        private: System::Windows::Forms::Label^  label4;
        private: System::Windows::Forms::Label^  label3;
        private: System::Windows::Forms::Label^  label2;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Training_PMD;

        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Training_Mean;

        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Training_Lower;

        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Training_Higher;

        private: System::Windows::Forms::Label^  label1;


        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Testing_Lower;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Testing_Higher;
        private: System::Windows::Forms::Label^  label17;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Testing_Current;


        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Validating_Lower;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Validating_Higher;
        private: System::Windows::Forms::Label^  label16;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Validating_Current;


        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Testing_Lower;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Testing_Higher;
        private: System::Windows::Forms::Label^  label15;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Testing_Current;


        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Validating_Lower;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Validating_Higher;
        private: System::Windows::Forms::Label^  label14;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Validating_Current;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Training_PMD;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Training_Mean;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Training_Lower;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Training_Higher;
        private: System::Windows::Forms::Label^  label13;
        private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Training_Current;
        private: System::Windows::Forms::Label^  label7;
private: System::Windows::Forms::Label^  Label_Accuracy_Mean;

        private: System::Windows::Forms::Label^  label9;
        private: System::Windows::Forms::Label^  label10;
        private: System::Windows::Forms::Label^  label11;
        private: System::Windows::Forms::Label^  label12;
private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Testing_PMD;

private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Testing_Mean;

private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Validating_PMD;

private: System::Windows::Forms::RichTextBox^  RichTextBox_Accuracy_Validating_Mean;

private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Testing_PMD;
private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Testing_Mean;
private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Validating_PMD;
private: System::Windows::Forms::RichTextBox^  RichTextBox_Loss_Validating_Mean;
private: System::Windows::Forms::Button^  Button_Panel_Chart_Grid_Search;

            ref struct struct_Chart_Parameters ^_ptr_chart_parameters_loss;
            ref struct struct_Chart_Parameters ^_ptr_chart_parameters_accuracy;
            ref struct struct_Chart_Parameters ^_ptr_chart_parameters_output;
private: System::Windows::Forms::DataGridView^  DataGridView_Grid_Search;
private: System::Windows::Forms::Button^  Button_Panel_Chart_Output;
private: System::Windows::Forms::Button^  Button_Chart_Output_Type;
private: System::Windows::Forms::Button^  Button_Chart_Output_Rescale;
private: System::Windows::Forms::Button^  Button_Chart_Loss_Difference;
private: System::Windows::Forms::Label^  Label_NumericUpDown_Chart_Loss_Y_Minimum;
private: System::Windows::Forms::Label^  Label_NumericUpDown_Chart_Loss_Y_Maximum;
private: System::Windows::Forms::NumericUpDown^  NumericUpDown_Chart_Loss_Y_Minimum;
private: System::Windows::Forms::NumericUpDown^  NumericUpDown_Chart_Loss_Y_Maximum;
private: System::Windows::Forms::Button^  Button_Chart_Loss_Apply;
private: System::Windows::Forms::Label^  Label_NumericUpDown_Chart_Accuracy_Y_Minimum;
private: System::Windows::Forms::Label^  Label_NumericUpDown_Chart_Accuracy_Y_Maximum;
private: System::Windows::Forms::NumericUpDown^  NumericUpDown_Chart_Accuracy_Y_Minimum;
private: System::Windows::Forms::NumericUpDown^  NumericUpDown_Chart_Accuracy_Y_Maximum;
private: System::Windows::Forms::Button^  Button_Chart_Accuracy_Apply;
private: System::Windows::Forms::Label^  Label_NumericUpDown_Chart_Output_Y_Minimum;
private: System::Windows::Forms::Label^  Label_NumericUpDown_Chart_Output_Y_Maximum;
private: System::Windows::Forms::NumericUpDown^  NumericUpDown_Chart_Output_Y_Minimum;
private: System::Windows::Forms::NumericUpDown^  NumericUpDown_Chart_Output_Y_Maximum;
private: System::Windows::Forms::Button^  Button_Chart_Output_Apply;

         ref struct struct_Chart_Parameters ^_ptr_chart_parameters_grid_search;
            System::Void Form_Neural_Network::ScaleView_Interval(System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received);
            System::Void Form_Neural_Network::ScaleView_Zoom(double const index_end_received,
                                                                                             ref struct struct_Chart_Parameters ^ptr_chart_parameters_received,
                                                                                             System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received);
            System::Void Form_Neural_Network::Compute_Precision(ref struct struct_Chart_Parameters ^ptr_chart_parameters_received, System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received);
            System::Void Chart_MouseWheel(ref struct struct_Chart_Parameters ^ptr_chart_parameters_received, System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received, MouseEventArgs^  e);
            System::Void Chart_Loss_MouseWheel(System::Object^  sender, MouseEventArgs^  e);
            System::Void Chart_Accuracy_MouseWheel(System::Object^  sender, MouseEventArgs^  e);
            System::Void Chart_Output_MouseWheel(System::Object^  sender, MouseEventArgs^  e);
            System::Void Chart_Grid_Search_MouseWheel(System::Object^  sender, MouseEventArgs^  e);
            System::Void Form_Neural_Network::Output_Loss_Information(enum MyEA::Common::ENUM_TYPE_DATASET const type_loss_received, unsigned int const index_series_received);
            System::Void Form_Neural_Network::Output_Accuracy_Information(enum MyEA::Common::ENUM_TYPE_DATASET const type_accuracy_received, unsigned int const index_series_received);

        private: System::Windows::Forms::Button^  Button_Panel_Chart_Accuracy;
        private: System::Windows::Forms::Panel^  Panel_Chart_Accuracy;
        private: System::Windows::Forms::Button^  Button_Chart_Accuracy_Reset;
        private: System::Windows::Forms::DataVisualization::Charting::Chart^  Chart_Accuracy;

        protected:
            /// <summary>
            /// Clean up any resources being used.
            /// </summary>
            ~Form_Neural_Network(void);
        private: System::Windows::Forms::Button^  Button_Training_Stop;
        private: System::Windows::Forms::Panel^  Panel_Button;
        private: System::Windows::Forms::Panel^  Panel_Chart_Loss;
        private: System::Windows::Forms::Panel^  Panel_Chart_Output;
        private: System::Windows::Forms::Panel^  Panel_Chart_Grid_Search;
        private: System::Windows::Forms::Button^  Button_Panel_Chart_Loss;
        private: System::Windows::Forms::Button^  Button_Chart_Loss_Reset;
        private: System::Windows::Forms::Button^  Button_Chart_Output_Reset;
        private: System::Windows::Forms::Button^  Button_Chart_Grid_Search_Reset;
        private: System::Windows::Forms::DataVisualization::Charting::Chart^  Chart_Loss;
        private: System::Windows::Forms::DataVisualization::Charting::Chart^  Chart_Output;
        private: System::Windows::Forms::DataVisualization::Charting::Chart^  Chart_Grid_Search;
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
                System::Windows::Forms::DataVisualization::Charting::Title^  title2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
                System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea3 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
                System::Windows::Forms::DataVisualization::Charting::Title^  title3 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
                System::Windows::Forms::DataVisualization::Charting::ChartArea^  chartArea4 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
                System::Windows::Forms::DataVisualization::Charting::Title^  title4 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
                this->label1 = (gcnew System::Windows::Forms::Label());
                this->label2 = (gcnew System::Windows::Forms::Label());
                this->label3 = (gcnew System::Windows::Forms::Label());
                this->label4 = (gcnew System::Windows::Forms::Label());
                this->label6 = (gcnew System::Windows::Forms::Label());
                this->label7 = (gcnew System::Windows::Forms::Label());
                this->label9 = (gcnew System::Windows::Forms::Label());
                this->label10 = (gcnew System::Windows::Forms::Label());
                this->label11 = (gcnew System::Windows::Forms::Label());
                this->label12 = (gcnew System::Windows::Forms::Label());
                this->label13 = (gcnew System::Windows::Forms::Label());
                this->label14 = (gcnew System::Windows::Forms::Label());
                this->label15 = (gcnew System::Windows::Forms::Label());
                this->label16 = (gcnew System::Windows::Forms::Label());
                this->label17 = (gcnew System::Windows::Forms::Label());
                this->RichTextBox_DataPoint_Loss = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_DataPoint_Accuracy = (gcnew System::Windows::Forms::RichTextBox());
                this->Button_Panel_Information = (gcnew System::Windows::Forms::Button());
                this->Button_Panel_Chart_Loss = (gcnew System::Windows::Forms::Button());
                this->Button_Panel_Chart_Accuracy = (gcnew System::Windows::Forms::Button());
                this->Button_Panel_Chart_Output = (gcnew System::Windows::Forms::Button());
                this->Button_Panel_Chart_Grid_Search = (gcnew System::Windows::Forms::Button());
                this->Button_Training_Menu = (gcnew System::Windows::Forms::Button());
                this->Button_Training_Stop = (gcnew System::Windows::Forms::Button());
                this->Panel_Button = (gcnew System::Windows::Forms::Panel());
                this->Panel_Information = (gcnew System::Windows::Forms::Panel());
                this->RichTextBox_Accuracy_Testing_PMD = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Testing_Mean = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Validating_PMD = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Validating_Mean = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Testing_PMD = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Testing_Mean = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Validating_PMD = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Validating_Mean = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Testing_Lower = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Testing_Higher = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Testing_Current = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Validating_Lower = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Validating_Higher = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Validating_Current = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Testing_Lower = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Testing_Higher = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Testing_Current = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Validating_Lower = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Validating_Higher = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Validating_Current = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Training_PMD = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Training_Mean = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Training_Lower = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Training_Higher = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Accuracy_Training_Current = (gcnew System::Windows::Forms::RichTextBox());
                this->Label_Accuracy_Mean = (gcnew System::Windows::Forms::Label());
                this->Label_Loss_Mean = (gcnew System::Windows::Forms::Label());
                this->RichTextBox_Loss_Training_PMD = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Training_Mean = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Training_Lower = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Loss_Training_Higher = (gcnew System::Windows::Forms::RichTextBox());
                this->Label_Loss = (gcnew System::Windows::Forms::Label());
                this->RichTextBox_Loss_Training_Current = (gcnew System::Windows::Forms::RichTextBox());
                this->RichTextBox_Output = (gcnew System::Windows::Forms::RichTextBox());
                this->Panel_Chart_Loss = (gcnew System::Windows::Forms::Panel());
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum = (gcnew System::Windows::Forms::Label());
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum = (gcnew System::Windows::Forms::Label());
                this->NumericUpDown_Chart_Loss_Y_Minimum = (gcnew System::Windows::Forms::NumericUpDown());
                this->NumericUpDown_Chart_Loss_Y_Maximum = (gcnew System::Windows::Forms::NumericUpDown());
                this->Button_Chart_Loss_Apply = (gcnew System::Windows::Forms::Button());
                this->Button_Chart_Loss_Difference = (gcnew System::Windows::Forms::Button());
                this->Chart_Loss = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->Button_Chart_Loss_Reset = (gcnew System::Windows::Forms::Button());
                this->Panel_Chart_Accuracy = (gcnew System::Windows::Forms::Panel());
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum = (gcnew System::Windows::Forms::Label());
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum = (gcnew System::Windows::Forms::Label());
                this->NumericUpDown_Chart_Accuracy_Y_Minimum = (gcnew System::Windows::Forms::NumericUpDown());
                this->NumericUpDown_Chart_Accuracy_Y_Maximum = (gcnew System::Windows::Forms::NumericUpDown());
                this->Button_Chart_Accuracy_Apply = (gcnew System::Windows::Forms::Button());
                this->Button_Chart_Accuracy_Reset = (gcnew System::Windows::Forms::Button());
                this->Chart_Accuracy = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->Panel_Chart_Output = (gcnew System::Windows::Forms::Panel());
                this->Chart_Output = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->Label_NumericUpDown_Chart_Output_Y_Minimum = (gcnew System::Windows::Forms::Label());
                this->Label_NumericUpDown_Chart_Output_Y_Maximum = (gcnew System::Windows::Forms::Label());
                this->NumericUpDown_Chart_Output_Y_Minimum = (gcnew System::Windows::Forms::NumericUpDown());
                this->NumericUpDown_Chart_Output_Y_Maximum = (gcnew System::Windows::Forms::NumericUpDown());
                this->Button_Chart_Output_Apply = (gcnew System::Windows::Forms::Button());
                this->Button_Chart_Output_Type = (gcnew System::Windows::Forms::Button());
                this->Button_Chart_Output_Rescale = (gcnew System::Windows::Forms::Button());
                this->Button_Chart_Output_Reset = (gcnew System::Windows::Forms::Button());
                this->Panel_Chart_Grid_Search = (gcnew System::Windows::Forms::Panel());
                this->DataGridView_Grid_Search = (gcnew System::Windows::Forms::DataGridView());
                this->Chart_Grid_Search = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
                this->Button_Chart_Grid_Search_Reset = (gcnew System::Windows::Forms::Button());
                this->Panel_Button->SuspendLayout();
                this->Panel_Information->SuspendLayout();
                this->Panel_Chart_Loss->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Loss_Y_Minimum))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Loss_Y_Maximum))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Loss))->BeginInit();
                this->Panel_Chart_Accuracy->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Accuracy_Y_Minimum))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Accuracy_Y_Maximum))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Accuracy))->BeginInit();
                this->Panel_Chart_Output->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Output))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Output_Y_Minimum))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Output_Y_Maximum))->BeginInit();
                this->Panel_Chart_Grid_Search->SuspendLayout();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->DataGridView_Grid_Search))->BeginInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Grid_Search))->BeginInit();
                this->SuspendLayout();
                // 
                // label1
                // 
                this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label1->Location = System::Drawing::Point(0, 20);
                this->label1->Margin = System::Windows::Forms::Padding(0);
                this->label1->Name = L"label1";
                this->label1->Size = System::Drawing::Size(72, 20);
                this->label1->TabIndex = 3;
                this->label1->Text = L"Training:";
                // 
                // label2
                // 
                this->label2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label2->Location = System::Drawing::Point(74, 0);
                this->label2->Margin = System::Windows::Forms::Padding(0);
                this->label2->Name = L"label2";
                this->label2->Size = System::Drawing::Size(90, 20);
                this->label2->TabIndex = 9;
                this->label2->Text = L"Current";
                // 
                // label3
                // 
                this->label3->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label3->Location = System::Drawing::Point(174, 0);
                this->label3->Margin = System::Windows::Forms::Padding(0);
                this->label3->Name = L"label3";
                this->label3->Size = System::Drawing::Size(90, 20);
                this->label3->TabIndex = 10;
                this->label3->Text = L"Higher";
                // 
                // label4
                // 
                this->label4->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label4->Location = System::Drawing::Point(274, 0);
                this->label4->Margin = System::Windows::Forms::Padding(0);
                this->label4->Name = L"label4";
                this->label4->Size = System::Drawing::Size(90, 20);
                this->label4->TabIndex = 11;
                this->label4->Text = L"Lower";
                // 
                // label6
                // 
                this->label6->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label6->Location = System::Drawing::Point(474, 0);
                this->label6->Margin = System::Windows::Forms::Padding(0);
                this->label6->Name = L"label6";
                this->label6->Size = System::Drawing::Size(100, 20);
                this->label6->TabIndex = 13;
                this->label6->Text = L"Past mean diff";
                // 
                // label7
                // 
                this->label7->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label7->Location = System::Drawing::Point(477, 129);
                this->label7->Margin = System::Windows::Forms::Padding(0);
                this->label7->Name = L"label7";
                this->label7->Size = System::Drawing::Size(100, 20);
                this->label7->TabIndex = 19;
                this->label7->Text = L"Past mean diff";
                // 
                // label9
                // 
                this->label9->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                  static_cast<System::Byte>(0)));
                this->label9->Location = System::Drawing::Point(277, 129);
                this->label9->Margin = System::Windows::Forms::Padding(0);
                this->label9->Name = L"label9";
                this->label9->Size = System::Drawing::Size(90, 20);
                this->label9->TabIndex = 17;
                this->label9->Text = L"Lower";
                // 
                // label10
                // 
                this->label10->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label10->Location = System::Drawing::Point(177, 129);
                this->label10->Margin = System::Windows::Forms::Padding(0);
                this->label10->Name = L"label10";
                this->label10->Size = System::Drawing::Size(90, 20);
                this->label10->TabIndex = 16;
                this->label10->Text = L"Higher";
                // 
                // label11
                // 
                this->label11->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label11->Location = System::Drawing::Point(77, 129);
                this->label11->Margin = System::Windows::Forms::Padding(0);
                this->label11->Name = L"label11";
                this->label11->Size = System::Drawing::Size(90, 20);
                this->label11->TabIndex = 15;
                this->label11->Text = L"Current";
                // 
                // label12
                // 
                this->label12->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label12->Location = System::Drawing::Point(0, 129);
                this->label12->Margin = System::Windows::Forms::Padding(0);
                this->label12->Name = L"label12";
                this->label12->Size = System::Drawing::Size(72, 20);
                this->label12->TabIndex = 14;
                this->label12->Text = L"Accuracy";
                // 
                // label13
                // 
                this->label13->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label13->Location = System::Drawing::Point(0, 149);
                this->label13->Margin = System::Windows::Forms::Padding(0);
                this->label13->Name = L"label13";
                this->label13->Size = System::Drawing::Size(72, 20);
                this->label13->TabIndex = 21;
                this->label13->Text = L"Training:";
                // 
                // label14
                // 
                this->label14->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label14->Location = System::Drawing::Point(0, 169);
                this->label14->Margin = System::Windows::Forms::Padding(0);
                this->label14->Name = L"label14";
                this->label14->Size = System::Drawing::Size(72, 20);
                this->label14->TabIndex = 27;
                this->label14->Text = L"Validating:";
                // 
                // label15
                // 
                this->label15->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label15->Location = System::Drawing::Point(0, 189);
                this->label15->Margin = System::Windows::Forms::Padding(0);
                this->label15->Name = L"label15";
                this->label15->Size = System::Drawing::Size(72, 20);
                this->label15->TabIndex = 33;
                this->label15->Text = L"Testing:";
                // 
                // label16
                // 
                this->label16->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label16->Location = System::Drawing::Point(0, 40);
                this->label16->Margin = System::Windows::Forms::Padding(0);
                this->label16->Name = L"label16";
                this->label16->Size = System::Drawing::Size(72, 20);
                this->label16->TabIndex = 39;
                this->label16->Text = L"Validating:";
                // 
                // label17
                // 
                this->label17->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                   static_cast<System::Byte>(0)));
                this->label17->Location = System::Drawing::Point(0, 60);
                this->label17->Margin = System::Windows::Forms::Padding(0);
                this->label17->Name = L"label17";
                this->label17->Size = System::Drawing::Size(72, 20);
                this->label17->TabIndex = 45;
                this->label17->Text = L"Testing:";
                // 
                // RichTextBox_DataPoint_Loss
                // 
                this->RichTextBox_DataPoint_Loss->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                                                                                                              | System::Windows::Forms::AnchorStyles::Right));
                this->RichTextBox_DataPoint_Loss->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_DataPoint_Loss->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_DataPoint_Loss->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_DataPoint_Loss->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular,
                                                                                      System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_DataPoint_Loss->Location = System::Drawing::Point(0, 272);
                this->RichTextBox_DataPoint_Loss->MaxLength = 256;
                this->RichTextBox_DataPoint_Loss->Name = L"RichTextBox_DataPoint_Loss";
                this->RichTextBox_DataPoint_Loss->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_DataPoint_Loss->Size = System::Drawing::Size(168, 15);
                this->RichTextBox_DataPoint_Loss->TabIndex = 1;
                this->RichTextBox_DataPoint_Loss->Text = L"DataPoint loss: 0";
                // 
                // RichTextBox_DataPoint_Accuracy
                // 
                this->RichTextBox_DataPoint_Accuracy->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                                                                                                                  | System::Windows::Forms::AnchorStyles::Right));
                this->RichTextBox_DataPoint_Accuracy->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_DataPoint_Accuracy->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_DataPoint_Accuracy->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_DataPoint_Accuracy->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular,
                                                                                          System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_DataPoint_Accuracy->Location = System::Drawing::Point(167, 272);
                this->RichTextBox_DataPoint_Accuracy->MaxLength = 256;
                this->RichTextBox_DataPoint_Accuracy->Name = L"RichTextBox_DataPoint_Accuracy";
                this->RichTextBox_DataPoint_Accuracy->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_DataPoint_Accuracy->Size = System::Drawing::Size(168, 15);
                this->RichTextBox_DataPoint_Accuracy->TabIndex = 1;
                this->RichTextBox_DataPoint_Accuracy->Text = L"DataPoint accuracy: 0";
                // 
                // Button_Panel_Information
                // 
                this->Button_Panel_Information->Location = System::Drawing::Point(0, 0);
                this->Button_Panel_Information->Margin = System::Windows::Forms::Padding(0);
                this->Button_Panel_Information->Name = L"Button_Panel_Information";
                this->Button_Panel_Information->Size = System::Drawing::Size(80, 35);
                this->Button_Panel_Information->TabIndex = 0;
                this->Button_Panel_Information->Text = L"Information";
                this->Button_Panel_Information->UseVisualStyleBackColor = true;
                this->Button_Panel_Information->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Panel_Information_Click);
                // 
                // Button_Panel_Chart_Loss
                // 
                this->Button_Panel_Chart_Loss->Location = System::Drawing::Point(80, 0);
                this->Button_Panel_Chart_Loss->Margin = System::Windows::Forms::Padding(0);
                this->Button_Panel_Chart_Loss->Name = L"Button_Panel_Chart_Loss";
                this->Button_Panel_Chart_Loss->Size = System::Drawing::Size(80, 35);
                this->Button_Panel_Chart_Loss->TabIndex = 1;
                this->Button_Panel_Chart_Loss->Text = L"Chart Loss";
                this->Button_Panel_Chart_Loss->UseVisualStyleBackColor = true;
                this->Button_Panel_Chart_Loss->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Panel_Chart_Loss_Click);
                // 
                // Button_Panel_Chart_Accuracy
                // 
                this->Button_Panel_Chart_Accuracy->Location = System::Drawing::Point(160, 0);
                this->Button_Panel_Chart_Accuracy->Margin = System::Windows::Forms::Padding(0);
                this->Button_Panel_Chart_Accuracy->Name = L"Button_Panel_Chart_Accuracy";
                this->Button_Panel_Chart_Accuracy->Size = System::Drawing::Size(80, 35);
                this->Button_Panel_Chart_Accuracy->TabIndex = 2;
                this->Button_Panel_Chart_Accuracy->Text = L"Chart Accuracy";
                this->Button_Panel_Chart_Accuracy->UseVisualStyleBackColor = true;
                this->Button_Panel_Chart_Accuracy->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Panel_Chart_Accuracy_Click);
                // 
                // Button_Panel_Chart_Output
                // 
                this->Button_Panel_Chart_Output->Location = System::Drawing::Point(240, 0);
                this->Button_Panel_Chart_Output->Margin = System::Windows::Forms::Padding(0);
                this->Button_Panel_Chart_Output->Name = L"Button_Panel_Chart_Output";
                this->Button_Panel_Chart_Output->Size = System::Drawing::Size(80, 35);
                this->Button_Panel_Chart_Output->TabIndex = 3;
                this->Button_Panel_Chart_Output->Text = L"Chart Output";
                this->Button_Panel_Chart_Output->UseVisualStyleBackColor = true;
                this->Button_Panel_Chart_Output->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Panel_Chart_Output_Click);
                // 
                // Button_Panel_Chart_Grid_Search
                // 
                this->Button_Panel_Chart_Grid_Search->Location = System::Drawing::Point(320, 0);
                this->Button_Panel_Chart_Grid_Search->Margin = System::Windows::Forms::Padding(0);
                this->Button_Panel_Chart_Grid_Search->Name = L"Button_Panel_Chart_Grid_Search";
                this->Button_Panel_Chart_Grid_Search->Size = System::Drawing::Size(80, 35);
                this->Button_Panel_Chart_Grid_Search->TabIndex = 4;
                this->Button_Panel_Chart_Grid_Search->Text = L"Chart Grid Search";
                this->Button_Panel_Chart_Grid_Search->UseVisualStyleBackColor = true;
                this->Button_Panel_Chart_Grid_Search->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Panel_Chart_Grid_Search_Click);
                // 
                // Button_Training_Menu
                // 
                this->Button_Training_Menu->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Training_Menu->Location = System::Drawing::Point(550, 0);
                this->Button_Training_Menu->Margin = System::Windows::Forms::Padding(0);
                this->Button_Training_Menu->Name = L"Button_Training_Menu";
                this->Button_Training_Menu->Size = System::Drawing::Size(60, 35);
                this->Button_Training_Menu->TabIndex = 5;
                this->Button_Training_Menu->Text = L"Menu";
                this->Button_Training_Menu->UseVisualStyleBackColor = true;
                this->Button_Training_Menu->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Training_Menu_Click);
                // 
                // Button_Training_Stop
                // 
                this->Button_Training_Stop->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Training_Stop->Location = System::Drawing::Point(610, 0);
                this->Button_Training_Stop->Margin = System::Windows::Forms::Padding(0);
                this->Button_Training_Stop->Name = L"Button_Training_Stop";
                this->Button_Training_Stop->Size = System::Drawing::Size(60, 35);
                this->Button_Training_Stop->TabIndex = 6;
                this->Button_Training_Stop->Text = L"Stop";
                this->Button_Training_Stop->UseVisualStyleBackColor = true;
                this->Button_Training_Stop->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Training_Stop_Click);
                // 
                // Panel_Button
                // 
                this->Panel_Button->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left)
                                                                                                | System::Windows::Forms::AnchorStyles::Right));
                this->Panel_Button->Controls->Add(this->Button_Panel_Chart_Output);
                this->Panel_Button->Controls->Add(this->Button_Panel_Chart_Grid_Search);
                this->Panel_Button->Controls->Add(this->Button_Panel_Information);
                this->Panel_Button->Controls->Add(this->Button_Training_Menu);
                this->Panel_Button->Controls->Add(this->Button_Panel_Chart_Accuracy);
                this->Panel_Button->Controls->Add(this->Button_Training_Stop);
                this->Panel_Button->Controls->Add(this->Button_Panel_Chart_Loss);
                this->Panel_Button->Location = System::Drawing::Point(0, 0);
                this->Panel_Button->Name = L"Panel_Button";
                this->Panel_Button->Size = System::Drawing::Size(670, 35);
                this->Panel_Button->TabIndex = 2;
                // 
                // Panel_Information
                // 
                this->Panel_Information->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                      | System::Windows::Forms::AnchorStyles::Left)
                                                                                                     | System::Windows::Forms::AnchorStyles::Right));
                this->Panel_Information->BackColor = System::Drawing::SystemColors::Window;
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Testing_PMD);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Testing_Mean);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Validating_PMD);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Validating_Mean);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Testing_PMD);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Testing_Mean);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Validating_PMD);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Validating_Mean);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Testing_Lower);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Testing_Higher);
                this->Panel_Information->Controls->Add(this->label17);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Testing_Current);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Validating_Lower);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Validating_Higher);
                this->Panel_Information->Controls->Add(this->label16);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Validating_Current);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Testing_Lower);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Testing_Higher);
                this->Panel_Information->Controls->Add(this->label15);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Testing_Current);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Validating_Lower);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Validating_Higher);
                this->Panel_Information->Controls->Add(this->label14);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Validating_Current);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Training_PMD);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Training_Mean);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Training_Lower);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Training_Higher);
                this->Panel_Information->Controls->Add(this->label13);
                this->Panel_Information->Controls->Add(this->RichTextBox_Accuracy_Training_Current);
                this->Panel_Information->Controls->Add(this->label7);
                this->Panel_Information->Controls->Add(this->Label_Accuracy_Mean);
                this->Panel_Information->Controls->Add(this->label9);
                this->Panel_Information->Controls->Add(this->label10);
                this->Panel_Information->Controls->Add(this->label11);
                this->Panel_Information->Controls->Add(this->label12);
                this->Panel_Information->Controls->Add(this->label6);
                this->Panel_Information->Controls->Add(this->Label_Loss_Mean);
                this->Panel_Information->Controls->Add(this->label4);
                this->Panel_Information->Controls->Add(this->label3);
                this->Panel_Information->Controls->Add(this->label2);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Training_PMD);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Training_Mean);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Training_Lower);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Training_Higher);
                this->Panel_Information->Controls->Add(this->label1);
                this->Panel_Information->Controls->Add(this->Label_Loss);
                this->Panel_Information->Controls->Add(this->RichTextBox_DataPoint_Accuracy);
                this->Panel_Information->Controls->Add(this->RichTextBox_DataPoint_Loss);
                this->Panel_Information->Controls->Add(this->RichTextBox_Loss_Training_Current);
                this->Panel_Information->Controls->Add(this->RichTextBox_Output);
                this->Panel_Information->Location = System::Drawing::Point(0, 38);
                this->Panel_Information->Name = L"Panel_Information";
                this->Panel_Information->Size = System::Drawing::Size(670, 415);
                this->Panel_Information->TabIndex = 3;
                // 
                // RichTextBox_Accuracy_Testing_PMD
                // 
                this->RichTextBox_Accuracy_Testing_PMD->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Testing_PMD->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Testing_PMD->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Testing_PMD->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                            System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Testing_PMD->Location = System::Drawing::Point(477, 189);
                this->RichTextBox_Accuracy_Testing_PMD->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Testing_PMD->MaxLength = 256;
                this->RichTextBox_Accuracy_Testing_PMD->Name = L"RichTextBox_Accuracy_Testing_PMD";
                this->RichTextBox_Accuracy_Testing_PMD->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Testing_PMD->Size = System::Drawing::Size(100, 20);
                this->RichTextBox_Accuracy_Testing_PMD->TabIndex = 55;
                this->RichTextBox_Accuracy_Testing_PMD->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Testing_Mean
                // 
                this->RichTextBox_Accuracy_Testing_Mean->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Testing_Mean->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Testing_Mean->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Testing_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                             System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Testing_Mean->Location = System::Drawing::Point(377, 189);
                this->RichTextBox_Accuracy_Testing_Mean->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Testing_Mean->MaxLength = 256;
                this->RichTextBox_Accuracy_Testing_Mean->Name = L"RichTextBox_Accuracy_Testing_Mean";
                this->RichTextBox_Accuracy_Testing_Mean->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Testing_Mean->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Testing_Mean->TabIndex = 54;
                this->RichTextBox_Accuracy_Testing_Mean->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Validating_PMD
                // 
                this->RichTextBox_Accuracy_Validating_PMD->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Validating_PMD->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Validating_PMD->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Validating_PMD->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                               System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Validating_PMD->Location = System::Drawing::Point(477, 169);
                this->RichTextBox_Accuracy_Validating_PMD->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Validating_PMD->MaxLength = 256;
                this->RichTextBox_Accuracy_Validating_PMD->Name = L"RichTextBox_Accuracy_Validating_PMD";
                this->RichTextBox_Accuracy_Validating_PMD->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Validating_PMD->Size = System::Drawing::Size(100, 20);
                this->RichTextBox_Accuracy_Validating_PMD->TabIndex = 53;
                this->RichTextBox_Accuracy_Validating_PMD->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Validating_Mean
                // 
                this->RichTextBox_Accuracy_Validating_Mean->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Validating_Mean->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Validating_Mean->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Validating_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Validating_Mean->Location = System::Drawing::Point(377, 169);
                this->RichTextBox_Accuracy_Validating_Mean->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Validating_Mean->MaxLength = 256;
                this->RichTextBox_Accuracy_Validating_Mean->Name = L"RichTextBox_Accuracy_Validating_Mean";
                this->RichTextBox_Accuracy_Validating_Mean->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Validating_Mean->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Validating_Mean->TabIndex = 52;
                this->RichTextBox_Accuracy_Validating_Mean->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Testing_PMD
                // 
                this->RichTextBox_Loss_Testing_PMD->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Testing_PMD->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Testing_PMD->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Testing_PMD->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                        System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Testing_PMD->Location = System::Drawing::Point(474, 60);
                this->RichTextBox_Loss_Testing_PMD->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Testing_PMD->MaxLength = 256;
                this->RichTextBox_Loss_Testing_PMD->Name = L"RichTextBox_Loss_Testing_PMD";
                this->RichTextBox_Loss_Testing_PMD->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Testing_PMD->Size = System::Drawing::Size(100, 20);
                this->RichTextBox_Loss_Testing_PMD->TabIndex = 51;
                this->RichTextBox_Loss_Testing_PMD->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Testing_Mean
                // 
                this->RichTextBox_Loss_Testing_Mean->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Testing_Mean->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Testing_Mean->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Testing_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                         System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Testing_Mean->Location = System::Drawing::Point(374, 60);
                this->RichTextBox_Loss_Testing_Mean->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Testing_Mean->MaxLength = 256;
                this->RichTextBox_Loss_Testing_Mean->Name = L"RichTextBox_Loss_Testing_Mean";
                this->RichTextBox_Loss_Testing_Mean->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Testing_Mean->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Testing_Mean->TabIndex = 50;
                this->RichTextBox_Loss_Testing_Mean->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Validating_PMD
                // 
                this->RichTextBox_Loss_Validating_PMD->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Validating_PMD->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Validating_PMD->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Validating_PMD->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                           System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Validating_PMD->Location = System::Drawing::Point(474, 40);
                this->RichTextBox_Loss_Validating_PMD->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Validating_PMD->MaxLength = 256;
                this->RichTextBox_Loss_Validating_PMD->Name = L"RichTextBox_Loss_Validating_PMD";
                this->RichTextBox_Loss_Validating_PMD->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Validating_PMD->Size = System::Drawing::Size(100, 20);
                this->RichTextBox_Loss_Validating_PMD->TabIndex = 49;
                this->RichTextBox_Loss_Validating_PMD->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Validating_Mean
                // 
                this->RichTextBox_Loss_Validating_Mean->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Validating_Mean->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Validating_Mean->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Validating_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                            System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Validating_Mean->Location = System::Drawing::Point(374, 40);
                this->RichTextBox_Loss_Validating_Mean->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Validating_Mean->MaxLength = 256;
                this->RichTextBox_Loss_Validating_Mean->Name = L"RichTextBox_Loss_Validating_Mean";
                this->RichTextBox_Loss_Validating_Mean->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Validating_Mean->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Validating_Mean->TabIndex = 48;
                this->RichTextBox_Loss_Validating_Mean->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Testing_Lower
                // 
                this->RichTextBox_Loss_Testing_Lower->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Testing_Lower->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Testing_Lower->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Testing_Lower->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                          System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Testing_Lower->Location = System::Drawing::Point(274, 60);
                this->RichTextBox_Loss_Testing_Lower->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Testing_Lower->MaxLength = 256;
                this->RichTextBox_Loss_Testing_Lower->Name = L"RichTextBox_Loss_Testing_Lower";
                this->RichTextBox_Loss_Testing_Lower->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Testing_Lower->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Testing_Lower->TabIndex = 47;
                this->RichTextBox_Loss_Testing_Lower->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Testing_Higher
                // 
                this->RichTextBox_Loss_Testing_Higher->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Testing_Higher->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Testing_Higher->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Testing_Higher->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                           System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Testing_Higher->Location = System::Drawing::Point(174, 60);
                this->RichTextBox_Loss_Testing_Higher->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Testing_Higher->MaxLength = 256;
                this->RichTextBox_Loss_Testing_Higher->Name = L"RichTextBox_Loss_Testing_Higher";
                this->RichTextBox_Loss_Testing_Higher->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Testing_Higher->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Testing_Higher->TabIndex = 46;
                this->RichTextBox_Loss_Testing_Higher->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Testing_Current
                // 
                this->RichTextBox_Loss_Testing_Current->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Testing_Current->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Testing_Current->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Testing_Current->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                            System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Testing_Current->Location = System::Drawing::Point(74, 60);
                this->RichTextBox_Loss_Testing_Current->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Testing_Current->MaxLength = 256;
                this->RichTextBox_Loss_Testing_Current->Name = L"RichTextBox_Loss_Testing_Current";
                this->RichTextBox_Loss_Testing_Current->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Testing_Current->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Testing_Current->TabIndex = 44;
                this->RichTextBox_Loss_Testing_Current->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Validating_Lower
                // 
                this->RichTextBox_Loss_Validating_Lower->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Validating_Lower->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Validating_Lower->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Validating_Lower->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                             System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Validating_Lower->Location = System::Drawing::Point(274, 40);
                this->RichTextBox_Loss_Validating_Lower->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Validating_Lower->MaxLength = 256;
                this->RichTextBox_Loss_Validating_Lower->Name = L"RichTextBox_Loss_Validating_Lower";
                this->RichTextBox_Loss_Validating_Lower->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Validating_Lower->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Validating_Lower->TabIndex = 41;
                this->RichTextBox_Loss_Validating_Lower->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Validating_Higher
                // 
                this->RichTextBox_Loss_Validating_Higher->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Validating_Higher->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Validating_Higher->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Validating_Higher->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                              System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Validating_Higher->Location = System::Drawing::Point(174, 40);
                this->RichTextBox_Loss_Validating_Higher->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Validating_Higher->MaxLength = 256;
                this->RichTextBox_Loss_Validating_Higher->Name = L"RichTextBox_Loss_Validating_Higher";
                this->RichTextBox_Loss_Validating_Higher->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Validating_Higher->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Validating_Higher->TabIndex = 40;
                this->RichTextBox_Loss_Validating_Higher->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Validating_Current
                // 
                this->RichTextBox_Loss_Validating_Current->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Validating_Current->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Validating_Current->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Validating_Current->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                               System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Validating_Current->Location = System::Drawing::Point(74, 40);
                this->RichTextBox_Loss_Validating_Current->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Validating_Current->MaxLength = 256;
                this->RichTextBox_Loss_Validating_Current->Name = L"RichTextBox_Loss_Validating_Current";
                this->RichTextBox_Loss_Validating_Current->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Validating_Current->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Validating_Current->TabIndex = 38;
                this->RichTextBox_Loss_Validating_Current->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Testing_Lower
                // 
                this->RichTextBox_Accuracy_Testing_Lower->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Testing_Lower->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Testing_Lower->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Testing_Lower->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                              System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Testing_Lower->Location = System::Drawing::Point(277, 189);
                this->RichTextBox_Accuracy_Testing_Lower->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Testing_Lower->MaxLength = 256;
                this->RichTextBox_Accuracy_Testing_Lower->Name = L"RichTextBox_Accuracy_Testing_Lower";
                this->RichTextBox_Accuracy_Testing_Lower->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Testing_Lower->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Testing_Lower->TabIndex = 35;
                this->RichTextBox_Accuracy_Testing_Lower->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Testing_Higher
                // 
                this->RichTextBox_Accuracy_Testing_Higher->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Testing_Higher->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Testing_Higher->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Testing_Higher->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                               System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Testing_Higher->Location = System::Drawing::Point(177, 189);
                this->RichTextBox_Accuracy_Testing_Higher->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Testing_Higher->MaxLength = 256;
                this->RichTextBox_Accuracy_Testing_Higher->Name = L"RichTextBox_Accuracy_Testing_Higher";
                this->RichTextBox_Accuracy_Testing_Higher->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Testing_Higher->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Testing_Higher->TabIndex = 34;
                this->RichTextBox_Accuracy_Testing_Higher->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Testing_Current
                // 
                this->RichTextBox_Accuracy_Testing_Current->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Testing_Current->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Testing_Current->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Testing_Current->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Testing_Current->Location = System::Drawing::Point(77, 189);
                this->RichTextBox_Accuracy_Testing_Current->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Testing_Current->MaxLength = 256;
                this->RichTextBox_Accuracy_Testing_Current->Name = L"RichTextBox_Accuracy_Testing_Current";
                this->RichTextBox_Accuracy_Testing_Current->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Testing_Current->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Testing_Current->TabIndex = 32;
                this->RichTextBox_Accuracy_Testing_Current->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Validating_Lower
                // 
                this->RichTextBox_Accuracy_Validating_Lower->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Validating_Lower->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Validating_Lower->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Validating_Lower->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                 System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Validating_Lower->Location = System::Drawing::Point(277, 169);
                this->RichTextBox_Accuracy_Validating_Lower->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Validating_Lower->MaxLength = 256;
                this->RichTextBox_Accuracy_Validating_Lower->Name = L"RichTextBox_Accuracy_Validating_Lower";
                this->RichTextBox_Accuracy_Validating_Lower->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Validating_Lower->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Validating_Lower->TabIndex = 29;
                this->RichTextBox_Accuracy_Validating_Lower->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Validating_Higher
                // 
                this->RichTextBox_Accuracy_Validating_Higher->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Validating_Higher->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Validating_Higher->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Validating_Higher->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                  System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Validating_Higher->Location = System::Drawing::Point(177, 169);
                this->RichTextBox_Accuracy_Validating_Higher->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Validating_Higher->MaxLength = 256;
                this->RichTextBox_Accuracy_Validating_Higher->Name = L"RichTextBox_Accuracy_Validating_Higher";
                this->RichTextBox_Accuracy_Validating_Higher->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Validating_Higher->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Validating_Higher->TabIndex = 28;
                this->RichTextBox_Accuracy_Validating_Higher->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Validating_Current
                // 
                this->RichTextBox_Accuracy_Validating_Current->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Validating_Current->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Validating_Current->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Validating_Current->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                   System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Validating_Current->Location = System::Drawing::Point(77, 169);
                this->RichTextBox_Accuracy_Validating_Current->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Validating_Current->MaxLength = 256;
                this->RichTextBox_Accuracy_Validating_Current->Name = L"RichTextBox_Accuracy_Validating_Current";
                this->RichTextBox_Accuracy_Validating_Current->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Validating_Current->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Validating_Current->TabIndex = 26;
                this->RichTextBox_Accuracy_Validating_Current->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Training_PMD
                // 
                this->RichTextBox_Accuracy_Training_PMD->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Training_PMD->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Training_PMD->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Training_PMD->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                             System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Training_PMD->Location = System::Drawing::Point(477, 149);
                this->RichTextBox_Accuracy_Training_PMD->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Training_PMD->MaxLength = 256;
                this->RichTextBox_Accuracy_Training_PMD->Name = L"RichTextBox_Accuracy_Training_PMD";
                this->RichTextBox_Accuracy_Training_PMD->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Training_PMD->Size = System::Drawing::Size(100, 20);
                this->RichTextBox_Accuracy_Training_PMD->TabIndex = 25;
                this->RichTextBox_Accuracy_Training_PMD->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Training_Mean
                // 
                this->RichTextBox_Accuracy_Training_Mean->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Training_Mean->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Training_Mean->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Training_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                              System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Training_Mean->Location = System::Drawing::Point(377, 149);
                this->RichTextBox_Accuracy_Training_Mean->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Training_Mean->MaxLength = 256;
                this->RichTextBox_Accuracy_Training_Mean->Name = L"RichTextBox_Accuracy_Training_Mean";
                this->RichTextBox_Accuracy_Training_Mean->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Training_Mean->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Training_Mean->TabIndex = 24;
                this->RichTextBox_Accuracy_Training_Mean->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Training_Lower
                // 
                this->RichTextBox_Accuracy_Training_Lower->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Training_Lower->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Training_Lower->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Training_Lower->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                               System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Training_Lower->Location = System::Drawing::Point(277, 149);
                this->RichTextBox_Accuracy_Training_Lower->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Training_Lower->MaxLength = 256;
                this->RichTextBox_Accuracy_Training_Lower->Name = L"RichTextBox_Accuracy_Training_Lower";
                this->RichTextBox_Accuracy_Training_Lower->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Training_Lower->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Training_Lower->TabIndex = 23;
                this->RichTextBox_Accuracy_Training_Lower->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Training_Higher
                // 
                this->RichTextBox_Accuracy_Training_Higher->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Training_Higher->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Training_Higher->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Training_Higher->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Training_Higher->Location = System::Drawing::Point(177, 149);
                this->RichTextBox_Accuracy_Training_Higher->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Training_Higher->MaxLength = 256;
                this->RichTextBox_Accuracy_Training_Higher->Name = L"RichTextBox_Accuracy_Training_Higher";
                this->RichTextBox_Accuracy_Training_Higher->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Training_Higher->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Training_Higher->TabIndex = 22;
                this->RichTextBox_Accuracy_Training_Higher->Text = L"000.000000000";
                // 
                // RichTextBox_Accuracy_Training_Current
                // 
                this->RichTextBox_Accuracy_Training_Current->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Accuracy_Training_Current->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Accuracy_Training_Current->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Accuracy_Training_Current->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                                 System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Accuracy_Training_Current->Location = System::Drawing::Point(77, 149);
                this->RichTextBox_Accuracy_Training_Current->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Accuracy_Training_Current->MaxLength = 256;
                this->RichTextBox_Accuracy_Training_Current->Name = L"RichTextBox_Accuracy_Training_Current";
                this->RichTextBox_Accuracy_Training_Current->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Accuracy_Training_Current->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Accuracy_Training_Current->TabIndex = 20;
                this->RichTextBox_Accuracy_Training_Current->Text = L"000.000000000";
                // 
                // Label_Accuracy_Mean
                // 
                this->Label_Accuracy_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular,
                                                                               System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->Label_Accuracy_Mean->Location = System::Drawing::Point(377, 129);
                this->Label_Accuracy_Mean->Margin = System::Windows::Forms::Padding(0);
                this->Label_Accuracy_Mean->Name = L"Label_Accuracy_Mean";
                this->Label_Accuracy_Mean->Size = System::Drawing::Size(90, 20);
                this->Label_Accuracy_Mean->TabIndex = 18;
                this->Label_Accuracy_Mean->Text = L"Mean";
                // 
                // Label_Loss_Mean
                // 
                this->Label_Loss_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                           static_cast<System::Byte>(0)));
                this->Label_Loss_Mean->Location = System::Drawing::Point(374, 0);
                this->Label_Loss_Mean->Margin = System::Windows::Forms::Padding(0);
                this->Label_Loss_Mean->Name = L"Label_Loss_Mean";
                this->Label_Loss_Mean->Size = System::Drawing::Size(90, 20);
                this->Label_Loss_Mean->TabIndex = 12;
                this->Label_Loss_Mean->Text = L"Mean";
                // 
                // RichTextBox_Loss_Training_PMD
                // 
                this->RichTextBox_Loss_Training_PMD->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Training_PMD->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Training_PMD->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Training_PMD->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                         System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Training_PMD->Location = System::Drawing::Point(474, 20);
                this->RichTextBox_Loss_Training_PMD->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Training_PMD->MaxLength = 256;
                this->RichTextBox_Loss_Training_PMD->Name = L"RichTextBox_Loss_Training_PMD";
                this->RichTextBox_Loss_Training_PMD->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Training_PMD->Size = System::Drawing::Size(100, 20);
                this->RichTextBox_Loss_Training_PMD->TabIndex = 7;
                this->RichTextBox_Loss_Training_PMD->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Training_Mean
                // 
                this->RichTextBox_Loss_Training_Mean->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Training_Mean->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Training_Mean->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Training_Mean->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                          System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Training_Mean->Location = System::Drawing::Point(374, 20);
                this->RichTextBox_Loss_Training_Mean->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Training_Mean->MaxLength = 256;
                this->RichTextBox_Loss_Training_Mean->Name = L"RichTextBox_Loss_Training_Mean";
                this->RichTextBox_Loss_Training_Mean->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Training_Mean->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Training_Mean->TabIndex = 6;
                this->RichTextBox_Loss_Training_Mean->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Training_Lower
                // 
                this->RichTextBox_Loss_Training_Lower->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Training_Lower->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Training_Lower->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Training_Lower->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                           System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Training_Lower->Location = System::Drawing::Point(274, 20);
                this->RichTextBox_Loss_Training_Lower->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Training_Lower->MaxLength = 256;
                this->RichTextBox_Loss_Training_Lower->Name = L"RichTextBox_Loss_Training_Lower";
                this->RichTextBox_Loss_Training_Lower->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Training_Lower->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Training_Lower->TabIndex = 5;
                this->RichTextBox_Loss_Training_Lower->Text = L"000.000000000";
                // 
                // RichTextBox_Loss_Training_Higher
                // 
                this->RichTextBox_Loss_Training_Higher->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Training_Higher->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Training_Higher->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Training_Higher->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                            System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Training_Higher->Location = System::Drawing::Point(174, 20);
                this->RichTextBox_Loss_Training_Higher->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Training_Higher->MaxLength = 256;
                this->RichTextBox_Loss_Training_Higher->Name = L"RichTextBox_Loss_Training_Higher";
                this->RichTextBox_Loss_Training_Higher->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Training_Higher->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Training_Higher->TabIndex = 4;
                this->RichTextBox_Loss_Training_Higher->Text = L"000.000000000";
                // 
                // Label_Loss
                // 
                this->Label_Loss->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                                      static_cast<System::Byte>(0)));
                this->Label_Loss->Location = System::Drawing::Point(0, 0);
                this->Label_Loss->Margin = System::Windows::Forms::Padding(0);
                this->Label_Loss->Name = L"Label_Loss";
                this->Label_Loss->Size = System::Drawing::Size(72, 20);
                this->Label_Loss->TabIndex = 2;
                this->Label_Loss->Text = L"Loss";
                // 
                // RichTextBox_Loss_Training_Current
                // 
                this->RichTextBox_Loss_Training_Current->BackColor = System::Drawing::SystemColors::Window;
                this->RichTextBox_Loss_Training_Current->BorderStyle = System::Windows::Forms::BorderStyle::None;
                this->RichTextBox_Loss_Training_Current->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Loss_Training_Current->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular,
                                                                                             System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
                this->RichTextBox_Loss_Training_Current->Location = System::Drawing::Point(74, 20);
                this->RichTextBox_Loss_Training_Current->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Loss_Training_Current->MaxLength = 256;
                this->RichTextBox_Loss_Training_Current->Name = L"RichTextBox_Loss_Training_Current";
                this->RichTextBox_Loss_Training_Current->ScrollBars = System::Windows::Forms::RichTextBoxScrollBars::None;
                this->RichTextBox_Loss_Training_Current->Size = System::Drawing::Size(90, 20);
                this->RichTextBox_Loss_Training_Current->TabIndex = 1;
                this->RichTextBox_Loss_Training_Current->Text = L"000.000000000";
                // 
                // RichTextBox_Output
                // 
                this->RichTextBox_Output->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                                                                                                      | System::Windows::Forms::AnchorStyles::Right));
                this->RichTextBox_Output->BackColor = System::Drawing::SystemColors::WindowFrame;
                this->RichTextBox_Output->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
                this->RichTextBox_Output->Cursor = System::Windows::Forms::Cursors::Default;
                this->RichTextBox_Output->Location = System::Drawing::Point(0, 289);
                this->RichTextBox_Output->Margin = System::Windows::Forms::Padding(0);
                this->RichTextBox_Output->Name = L"RichTextBox_Output";
                this->RichTextBox_Output->Size = System::Drawing::Size(673, 128);
                this->RichTextBox_Output->TabIndex = 0;
                this->RichTextBox_Output->Text = L"";
                // 
                // Panel_Chart_Loss
                // 
                this->Panel_Chart_Loss->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                     | System::Windows::Forms::AnchorStyles::Left)
                                                                                                    | System::Windows::Forms::AnchorStyles::Right));
                this->Panel_Chart_Loss->Controls->Add(this->Label_NumericUpDown_Chart_Loss_Y_Minimum);
                this->Panel_Chart_Loss->Controls->Add(this->Label_NumericUpDown_Chart_Loss_Y_Maximum);
                this->Panel_Chart_Loss->Controls->Add(this->NumericUpDown_Chart_Loss_Y_Minimum);
                this->Panel_Chart_Loss->Controls->Add(this->NumericUpDown_Chart_Loss_Y_Maximum);
                this->Panel_Chart_Loss->Controls->Add(this->Button_Chart_Loss_Apply);
                this->Panel_Chart_Loss->Controls->Add(this->Button_Chart_Loss_Difference);
                this->Panel_Chart_Loss->Controls->Add(this->Chart_Loss);
                this->Panel_Chart_Loss->Controls->Add(this->Button_Chart_Loss_Reset);
                this->Panel_Chart_Loss->Location = System::Drawing::Point(0, 38);
                this->Panel_Chart_Loss->Name = L"Panel_Chart_Loss";
                this->Panel_Chart_Loss->Size = System::Drawing::Size(670, 415);
                this->Panel_Chart_Loss->TabIndex = 0;
                this->Panel_Chart_Loss->Visible = false;
                // 
                // Label_NumericUpDown_Chart_Loss_Y_Minimum
                // 
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum->Location = System::Drawing::Point(3, 3);
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum->Name = L"Label_NumericUpDown_Chart_Loss_Y_Minimum";
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum->Size = System::Drawing::Size(40, 20);
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum->TabIndex = 6;
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum->Text = L"Y-Min:";
                this->Label_NumericUpDown_Chart_Loss_Y_Minimum->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
                // 
                // Label_NumericUpDown_Chart_Loss_Y_Maximum
                // 
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum->Location = System::Drawing::Point(120, 3);
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum->Name = L"Label_NumericUpDown_Chart_Loss_Y_Maximum";
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum->Size = System::Drawing::Size(40, 20);
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum->TabIndex = 6;
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum->Text = L"Y-Max:";
                this->Label_NumericUpDown_Chart_Loss_Y_Maximum->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
                // 
                // NumericUpDown_Chart_Loss_Y_Minimum
                // 
                this->NumericUpDown_Chart_Loss_Y_Minimum->DecimalPlaces = 7;
                this->NumericUpDown_Chart_Loss_Y_Minimum->Location = System::Drawing::Point(49, 3);
                this->NumericUpDown_Chart_Loss_Y_Minimum->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, System::Int32::MinValue });
                this->NumericUpDown_Chart_Loss_Y_Minimum->Name = L"NumericUpDown_Chart_Loss_Y_Minimum";
                this->NumericUpDown_Chart_Loss_Y_Minimum->Size = System::Drawing::Size(62, 20);
                this->NumericUpDown_Chart_Loss_Y_Minimum->TabIndex = 5;
                // 
                // NumericUpDown_Chart_Loss_Y_Maximum
                // 
                this->NumericUpDown_Chart_Loss_Y_Maximum->DecimalPlaces = 7;
                this->NumericUpDown_Chart_Loss_Y_Maximum->Location = System::Drawing::Point(163, 3);
                this->NumericUpDown_Chart_Loss_Y_Maximum->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, System::Int32::MinValue });
                this->NumericUpDown_Chart_Loss_Y_Maximum->Name = L"NumericUpDown_Chart_Loss_Y_Maximum";
                this->NumericUpDown_Chart_Loss_Y_Maximum->Size = System::Drawing::Size(62, 20);
                this->NumericUpDown_Chart_Loss_Y_Maximum->TabIndex = 5;
                // 
                // Button_Chart_Loss_Apply
                // 
                this->Button_Chart_Loss_Apply->Location = System::Drawing::Point(231, 3);
                this->Button_Chart_Loss_Apply->Name = L"Button_Chart_Loss_Apply";
                this->Button_Chart_Loss_Apply->Size = System::Drawing::Size(54, 20);
                this->Button_Chart_Loss_Apply->TabIndex = 4;
                this->Button_Chart_Loss_Apply->Text = L"Apply";
                this->Button_Chart_Loss_Apply->UseVisualStyleBackColor = true;
                this->Button_Chart_Loss_Apply->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Loss_Apply_Click);
                // 
                // Button_Chart_Loss_Difference
                // 
                this->Button_Chart_Loss_Difference->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Loss_Difference->Location = System::Drawing::Point(496, 3);
                this->Button_Chart_Loss_Difference->Name = L"Button_Chart_Loss_Difference";
                this->Button_Chart_Loss_Difference->Size = System::Drawing::Size(96, 23);
                this->Button_Chart_Loss_Difference->TabIndex = 3;
                this->Button_Chart_Loss_Difference->Text = L"Difference (false)";
                this->Button_Chart_Loss_Difference->UseVisualStyleBackColor = true;
                this->Button_Chart_Loss_Difference->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Loss_Difference_Click);
                // 
                // Chart_Loss
                // 
                this->Chart_Loss->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                               | System::Windows::Forms::AnchorStyles::Left)
                                                                                              | System::Windows::Forms::AnchorStyles::Right));
                chartArea1->AxisX->InterlacedColor = System::Drawing::Color::White;
                chartArea1->AxisX->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea1->AxisX->Maximum = 1;
                chartArea1->AxisX->Title = L"EPOCH(s)";
                chartArea1->AxisY->InterlacedColor = System::Drawing::Color::Black;
                chartArea1->AxisY->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea1->AxisY->ScaleBreakStyle->StartFromZero = System::Windows::Forms::DataVisualization::Charting::StartFromZero::Yes;
                chartArea1->AxisY->TextOrientation = System::Windows::Forms::DataVisualization::Charting::TextOrientation::Stacked;
                chartArea1->AxisY->Title = L"Loss";
                chartArea1->BackColor = System::Drawing::Color::White;
                chartArea1->BackSecondaryColor = System::Drawing::Color::White;
                chartArea1->Name = L"ChartArea_Loss";
                this->Chart_Loss->ChartAreas->Add(chartArea1);
                this->Chart_Loss->Location = System::Drawing::Point(0, 29);
                this->Chart_Loss->Margin = System::Windows::Forms::Padding(0);
                this->Chart_Loss->Name = L"Chart_Loss";
                this->Chart_Loss->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                this->Chart_Loss->Size = System::Drawing::Size(670, 386);
                this->Chart_Loss->TabIndex = 1;
                title1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                            static_cast<System::Byte>(0)));
                title1->Name = L"Chart_Title";
                title1->Text = L"Chart Loss";
                this->Chart_Loss->Titles->Add(title1);
                this->Chart_Loss->MouseEnter += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Loss_MouseEnter);
                this->Chart_Loss->MouseLeave += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Loss_MouseLeave);
                // 
                // Button_Chart_Loss_Reset
                // 
                this->Button_Chart_Loss_Reset->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Loss_Reset->Location = System::Drawing::Point(592, 3);
                this->Button_Chart_Loss_Reset->Name = L"Button_Chart_Loss_Reset";
                this->Button_Chart_Loss_Reset->Size = System::Drawing::Size(75, 23);
                this->Button_Chart_Loss_Reset->TabIndex = 2;
                this->Button_Chart_Loss_Reset->Text = L"Reset";
                this->Button_Chart_Loss_Reset->UseVisualStyleBackColor = true;
                this->Button_Chart_Loss_Reset->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Loss_Reset__Click);
                // 
                // Panel_Chart_Accuracy
                // 
                this->Panel_Chart_Accuracy->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                         | System::Windows::Forms::AnchorStyles::Left)
                                                                                                        | System::Windows::Forms::AnchorStyles::Right));
                this->Panel_Chart_Accuracy->Controls->Add(this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum);
                this->Panel_Chart_Accuracy->Controls->Add(this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum);
                this->Panel_Chart_Accuracy->Controls->Add(this->NumericUpDown_Chart_Accuracy_Y_Minimum);
                this->Panel_Chart_Accuracy->Controls->Add(this->NumericUpDown_Chart_Accuracy_Y_Maximum);
                this->Panel_Chart_Accuracy->Controls->Add(this->Button_Chart_Accuracy_Apply);
                this->Panel_Chart_Accuracy->Controls->Add(this->Button_Chart_Accuracy_Reset);
                this->Panel_Chart_Accuracy->Controls->Add(this->Chart_Accuracy);
                this->Panel_Chart_Accuracy->Location = System::Drawing::Point(0, 38);
                this->Panel_Chart_Accuracy->Name = L"Panel_Chart_Accuracy";
                this->Panel_Chart_Accuracy->Size = System::Drawing::Size(670, 415);
                this->Panel_Chart_Accuracy->TabIndex = 3;
                this->Panel_Chart_Accuracy->Visible = false;
                // 
                // Label_NumericUpDown_Chart_Accuracy_Y_Minimum
                // 
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum->Location = System::Drawing::Point(3, 3);
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum->Name = L"Label_NumericUpDown_Chart_Accuracy_Y_Minimum";
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum->Size = System::Drawing::Size(40, 20);
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum->TabIndex = 6;
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum->Text = L"Y-Min:";
                this->Label_NumericUpDown_Chart_Accuracy_Y_Minimum->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
                // 
                // Label_NumericUpDown_Chart_Accuracy_Y_Maximum
                // 
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum->Location = System::Drawing::Point(120, 3);
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum->Name = L"Label_NumericUpDown_Chart_Accuracy_Y_Maximum";
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum->Size = System::Drawing::Size(40, 20);
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum->TabIndex = 6;
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum->Text = L"Y-Max:";
                this->Label_NumericUpDown_Chart_Accuracy_Y_Maximum->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
                // 
                // NumericUpDown_Chart_Accuracy_Y_Minimum
                // 
                this->NumericUpDown_Chart_Accuracy_Y_Minimum->DecimalPlaces = 7;
                this->NumericUpDown_Chart_Accuracy_Y_Minimum->Location = System::Drawing::Point(49, 3);
                this->NumericUpDown_Chart_Accuracy_Y_Minimum->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) {
                    100, 0, 0,
                        System::Int32::MinValue
                });
                this->NumericUpDown_Chart_Accuracy_Y_Minimum->Name = L"NumericUpDown_Chart_Accuracy_Y_Minimum";
                this->NumericUpDown_Chart_Accuracy_Y_Minimum->Size = System::Drawing::Size(62, 20);
                this->NumericUpDown_Chart_Accuracy_Y_Minimum->TabIndex = 5;
                // 
                // NumericUpDown_Chart_Accuracy_Y_Maximum
                // 
                this->NumericUpDown_Chart_Accuracy_Y_Maximum->DecimalPlaces = 7;
                this->NumericUpDown_Chart_Accuracy_Y_Maximum->Location = System::Drawing::Point(163, 3);
                this->NumericUpDown_Chart_Accuracy_Y_Maximum->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) {
                    100, 0, 0,
                        System::Int32::MinValue
                });
                this->NumericUpDown_Chart_Accuracy_Y_Maximum->Name = L"NumericUpDown_Chart_Accuracy_Y_Maximum";
                this->NumericUpDown_Chart_Accuracy_Y_Maximum->Size = System::Drawing::Size(62, 20);
                this->NumericUpDown_Chart_Accuracy_Y_Maximum->TabIndex = 5;
                // 
                // Button_Chart_Accuracy_Apply
                // 
                this->Button_Chart_Accuracy_Apply->Location = System::Drawing::Point(231, 3);
                this->Button_Chart_Accuracy_Apply->Name = L"Button_Chart_Accuracy_Apply";
                this->Button_Chart_Accuracy_Apply->Size = System::Drawing::Size(54, 20);
                this->Button_Chart_Accuracy_Apply->TabIndex = 4;
                this->Button_Chart_Accuracy_Apply->Text = L"Apply";
                this->Button_Chart_Accuracy_Apply->UseVisualStyleBackColor = true;
                this->Button_Chart_Accuracy_Apply->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Accuracy_Apply_Click);
                // 
                // Button_Chart_Accuracy_Reset
                // 
                this->Button_Chart_Accuracy_Reset->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Accuracy_Reset->Location = System::Drawing::Point(592, 3);
                this->Button_Chart_Accuracy_Reset->Name = L"Button_Chart_Accuracy_Reset";
                this->Button_Chart_Accuracy_Reset->Size = System::Drawing::Size(75, 23);
                this->Button_Chart_Accuracy_Reset->TabIndex = 2;
                this->Button_Chart_Accuracy_Reset->Text = L"Reset";
                this->Button_Chart_Accuracy_Reset->UseVisualStyleBackColor = true;
                this->Button_Chart_Accuracy_Reset->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Accuracy_Reset__Click);
                // 
                // Chart_Accuracy
                // 
                this->Chart_Accuracy->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                   | System::Windows::Forms::AnchorStyles::Left)
                                                                                                  | System::Windows::Forms::AnchorStyles::Right));
                chartArea2->AxisX->InterlacedColor = System::Drawing::Color::White;
                chartArea2->AxisX->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea2->AxisX->Maximum = 1;
                chartArea2->AxisX->Title = L"EPOCH(s)";
                chartArea2->AxisY->InterlacedColor = System::Drawing::Color::Black;
                chartArea2->AxisY->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea2->AxisY->ScaleBreakStyle->StartFromZero = System::Windows::Forms::DataVisualization::Charting::StartFromZero::Yes;
                chartArea2->AxisY->TextOrientation = System::Windows::Forms::DataVisualization::Charting::TextOrientation::Stacked;
                chartArea2->AxisY->Title = L"Accuracy";
                chartArea2->BackColor = System::Drawing::Color::White;
                chartArea2->BackSecondaryColor = System::Drawing::Color::White;
                chartArea2->Name = L"ChartArea_Accuracy";
                this->Chart_Accuracy->ChartAreas->Add(chartArea2);
                this->Chart_Accuracy->Location = System::Drawing::Point(0, 29);
                this->Chart_Accuracy->Margin = System::Windows::Forms::Padding(0);
                this->Chart_Accuracy->Name = L"Chart_Accuracy";
                this->Chart_Accuracy->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                this->Chart_Accuracy->Size = System::Drawing::Size(670, 386);
                this->Chart_Accuracy->TabIndex = 1;
                title2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                            static_cast<System::Byte>(0)));
                title2->Name = L"Chart_Title";
                title2->Text = L"Chart Accuracy";
                this->Chart_Accuracy->Titles->Add(title2);
                this->Chart_Accuracy->MouseEnter += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Accuracy_MouseEnter);
                this->Chart_Accuracy->MouseLeave += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Accuracy_MouseLeave);
                // 
                // Panel_Chart_Output
                // 
                this->Panel_Chart_Output->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                       | System::Windows::Forms::AnchorStyles::Left)
                                                                                                      | System::Windows::Forms::AnchorStyles::Right));
                this->Panel_Chart_Output->Controls->Add(this->Chart_Output);
                this->Panel_Chart_Output->Controls->Add(this->Label_NumericUpDown_Chart_Output_Y_Minimum);
                this->Panel_Chart_Output->Controls->Add(this->Label_NumericUpDown_Chart_Output_Y_Maximum);
                this->Panel_Chart_Output->Controls->Add(this->NumericUpDown_Chart_Output_Y_Minimum);
                this->Panel_Chart_Output->Controls->Add(this->NumericUpDown_Chart_Output_Y_Maximum);
                this->Panel_Chart_Output->Controls->Add(this->Button_Chart_Output_Apply);
                this->Panel_Chart_Output->Controls->Add(this->Button_Chart_Output_Type);
                this->Panel_Chart_Output->Controls->Add(this->Button_Chart_Output_Rescale);
                this->Panel_Chart_Output->Controls->Add(this->Button_Chart_Output_Reset);
                this->Panel_Chart_Output->Location = System::Drawing::Point(0, 38);
                this->Panel_Chart_Output->Name = L"Panel_Chart_Output";
                this->Panel_Chart_Output->Size = System::Drawing::Size(670, 415);
                this->Panel_Chart_Output->TabIndex = 0;
                this->Panel_Chart_Output->Visible = false;
                // 
                // Chart_Output
                // 
                this->Chart_Output->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                 | System::Windows::Forms::AnchorStyles::Left)
                                                                                                | System::Windows::Forms::AnchorStyles::Right));
                chartArea3->AxisX->InterlacedColor = System::Drawing::Color::White;
                chartArea3->AxisX->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea3->AxisX->Title = L"Data";
                chartArea3->AxisY->InterlacedColor = System::Drawing::Color::Black;
                chartArea3->AxisY->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea3->AxisY->ScaleBreakStyle->StartFromZero = System::Windows::Forms::DataVisualization::Charting::StartFromZero::Yes;
                chartArea3->AxisY->TextOrientation = System::Windows::Forms::DataVisualization::Charting::TextOrientation::Stacked;
                chartArea3->AxisY->Title = L"Output";
                chartArea3->BackColor = System::Drawing::Color::White;
                chartArea3->BackSecondaryColor = System::Drawing::Color::White;
                chartArea3->Name = L"ChartArea_Output";
                this->Chart_Output->ChartAreas->Add(chartArea3);
                this->Chart_Output->Location = System::Drawing::Point(0, 29);
                this->Chart_Output->Margin = System::Windows::Forms::Padding(0);
                this->Chart_Output->Name = L"Chart_Output";
                this->Chart_Output->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                this->Chart_Output->Size = System::Drawing::Size(670, 386);
                this->Chart_Output->TabIndex = 1;
                title3->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                            static_cast<System::Byte>(0)));
                title3->Name = L"Chart_Title";
                title3->Text = L"Chart Output";
                this->Chart_Output->Titles->Add(title3);
                this->Chart_Output->MouseEnter += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Output_MouseEnter);
                this->Chart_Output->MouseLeave += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Output_MouseLeave);
                // 
                // Label_NumericUpDown_Chart_Output_Y_Minimum
                // 
                this->Label_NumericUpDown_Chart_Output_Y_Minimum->Location = System::Drawing::Point(3, 3);
                this->Label_NumericUpDown_Chart_Output_Y_Minimum->Name = L"Label_NumericUpDown_Chart_Output_Y_Minimum";
                this->Label_NumericUpDown_Chart_Output_Y_Minimum->Size = System::Drawing::Size(40, 20);
                this->Label_NumericUpDown_Chart_Output_Y_Minimum->TabIndex = 6;
                this->Label_NumericUpDown_Chart_Output_Y_Minimum->Text = L"Y-Min:";
                this->Label_NumericUpDown_Chart_Output_Y_Minimum->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
                // 
                // Label_NumericUpDown_Chart_Output_Y_Maximum
                // 
                this->Label_NumericUpDown_Chart_Output_Y_Maximum->Location = System::Drawing::Point(120, 3);
                this->Label_NumericUpDown_Chart_Output_Y_Maximum->Name = L"Label_NumericUpDown_Chart_Output_Y_Maximum";
                this->Label_NumericUpDown_Chart_Output_Y_Maximum->Size = System::Drawing::Size(40, 20);
                this->Label_NumericUpDown_Chart_Output_Y_Maximum->TabIndex = 6;
                this->Label_NumericUpDown_Chart_Output_Y_Maximum->Text = L"Y-Max:";
                this->Label_NumericUpDown_Chart_Output_Y_Maximum->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
                // 
                // NumericUpDown_Chart_Output_Y_Minimum
                // 
                this->NumericUpDown_Chart_Output_Y_Minimum->DecimalPlaces = 7;
                this->NumericUpDown_Chart_Output_Y_Minimum->Location = System::Drawing::Point(49, 3);
                this->NumericUpDown_Chart_Output_Y_Minimum->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
                this->NumericUpDown_Chart_Output_Y_Minimum->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, System::Int32::MinValue });
                this->NumericUpDown_Chart_Output_Y_Minimum->Name = L"NumericUpDown_Chart_Output_Y_Minimum";
                this->NumericUpDown_Chart_Output_Y_Minimum->Size = System::Drawing::Size(62, 20);
                this->NumericUpDown_Chart_Output_Y_Minimum->TabIndex = 5;
                // 
                // NumericUpDown_Chart_Output_Y_Maximum
                // 
                this->NumericUpDown_Chart_Output_Y_Maximum->DecimalPlaces = 7;
                this->NumericUpDown_Chart_Output_Y_Maximum->Location = System::Drawing::Point(163, 3);
                this->NumericUpDown_Chart_Output_Y_Maximum->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
                this->NumericUpDown_Chart_Output_Y_Maximum->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, System::Int32::MinValue });
                this->NumericUpDown_Chart_Output_Y_Maximum->Name = L"NumericUpDown_Chart_Output_Y_Maximum";
                this->NumericUpDown_Chart_Output_Y_Maximum->Size = System::Drawing::Size(62, 20);
                this->NumericUpDown_Chart_Output_Y_Maximum->TabIndex = 5;
                // 
                // Button_Chart_Output_Apply
                // 
                this->Button_Chart_Output_Apply->Location = System::Drawing::Point(231, 3);
                this->Button_Chart_Output_Apply->Name = L"Button_Chart_Output_Apply";
                this->Button_Chart_Output_Apply->Size = System::Drawing::Size(54, 20);
                this->Button_Chart_Output_Apply->TabIndex = 4;
                this->Button_Chart_Output_Apply->Text = L"Apply";
                this->Button_Chart_Output_Apply->UseVisualStyleBackColor = true;
                this->Button_Chart_Output_Apply->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Output_Apply_Click);
                // 
                // Button_Chart_Output_Type
                // 
                this->Button_Chart_Output_Type->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Output_Type->Location = System::Drawing::Point(422, 3);
                this->Button_Chart_Output_Type->Name = L"Button_Chart_Output_Type";
                this->Button_Chart_Output_Type->Size = System::Drawing::Size(95, 23);
                this->Button_Chart_Output_Type->TabIndex = 3;
                this->Button_Chart_Output_Type->Text = L"Fast Point";
                this->Button_Chart_Output_Type->UseVisualStyleBackColor = true;
                this->Button_Chart_Output_Type->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Output_Type_Click);
                // 
                // Button_Chart_Output_Rescale
                // 
                this->Button_Chart_Output_Rescale->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Output_Rescale->Location = System::Drawing::Point(517, 3);
                this->Button_Chart_Output_Rescale->Name = L"Button_Chart_Output_Rescale";
                this->Button_Chart_Output_Rescale->Size = System::Drawing::Size(75, 23);
                this->Button_Chart_Output_Rescale->TabIndex = 4;
                this->Button_Chart_Output_Rescale->Text = L"Rescale";
                this->Button_Chart_Output_Rescale->UseVisualStyleBackColor = true;
                this->Button_Chart_Output_Rescale->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Output_Rescale_Click);
                // 
                // Button_Chart_Output_Reset
                // 
                this->Button_Chart_Output_Reset->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Output_Reset->Location = System::Drawing::Point(592, 3);
                this->Button_Chart_Output_Reset->Name = L"Button_Chart_Output_Reset";
                this->Button_Chart_Output_Reset->Size = System::Drawing::Size(75, 23);
                this->Button_Chart_Output_Reset->TabIndex = 2;
                this->Button_Chart_Output_Reset->Text = L"Reset";
                this->Button_Chart_Output_Reset->UseVisualStyleBackColor = true;
                this->Button_Chart_Output_Reset->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Output_Reset__Click);
                // 
                // Panel_Chart_Grid_Search
                // 
                this->Panel_Chart_Grid_Search->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                            | System::Windows::Forms::AnchorStyles::Left)
                                                                                                           | System::Windows::Forms::AnchorStyles::Right));
                this->Panel_Chart_Grid_Search->Controls->Add(this->DataGridView_Grid_Search);
                this->Panel_Chart_Grid_Search->Controls->Add(this->Chart_Grid_Search);
                this->Panel_Chart_Grid_Search->Controls->Add(this->Button_Chart_Grid_Search_Reset);
                this->Panel_Chart_Grid_Search->Location = System::Drawing::Point(0, 38);
                this->Panel_Chart_Grid_Search->Name = L"Panel_Chart_Grid_Search";
                this->Panel_Chart_Grid_Search->Size = System::Drawing::Size(670, 415);
                this->Panel_Chart_Grid_Search->TabIndex = 0;
                this->Panel_Chart_Grid_Search->Visible = false;
                // 
                // DataGridView_Grid_Search
                // 
                this->DataGridView_Grid_Search->AllowUserToAddRows = false;
                this->DataGridView_Grid_Search->AllowUserToDeleteRows = false;
                this->DataGridView_Grid_Search->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                                                                                                            | System::Windows::Forms::AnchorStyles::Right));
                this->DataGridView_Grid_Search->AutoSizeColumnsMode = System::Windows::Forms::DataGridViewAutoSizeColumnsMode::DisplayedCells;
                this->DataGridView_Grid_Search->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
                this->DataGridView_Grid_Search->EditMode = System::Windows::Forms::DataGridViewEditMode::EditProgrammatically;
                this->DataGridView_Grid_Search->Location = System::Drawing::Point(4, 313);
                this->DataGridView_Grid_Search->MultiSelect = false;
                this->DataGridView_Grid_Search->Name = L"DataGridView_Grid_Search";
                this->DataGridView_Grid_Search->ReadOnly = true;
                this->DataGridView_Grid_Search->RowHeadersWidthSizeMode = System::Windows::Forms::DataGridViewRowHeadersWidthSizeMode::AutoSizeToAllHeaders;
                this->DataGridView_Grid_Search->ShowCellErrors = false;
                this->DataGridView_Grid_Search->ShowEditingIcon = false;
                this->DataGridView_Grid_Search->ShowRowErrors = false;
                this->DataGridView_Grid_Search->Size = System::Drawing::Size(663, 99);
                this->DataGridView_Grid_Search->TabIndex = 3;
                // 
                // Chart_Grid_Search
                // 
                this->Chart_Grid_Search->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                                                                                                      | System::Windows::Forms::AnchorStyles::Left)
                                                                                                     | System::Windows::Forms::AnchorStyles::Right));
                chartArea4->AxisX->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea4->AxisX->Maximum = 1;
                chartArea4->AxisY->MajorGrid->LineColor = System::Drawing::Color::Gainsboro;
                chartArea4->Name = L"ChartArea_Grid_Search";
                this->Chart_Grid_Search->ChartAreas->Add(chartArea4);
                this->Chart_Grid_Search->Location = System::Drawing::Point(0, 29);
                this->Chart_Grid_Search->Margin = System::Windows::Forms::Padding(0);
                this->Chart_Grid_Search->Name = L"Chart_Grid_Search";
                this->Chart_Grid_Search->Palette = System::Windows::Forms::DataVisualization::Charting::ChartColorPalette::Bright;
                this->Chart_Grid_Search->Size = System::Drawing::Size(670, 280);
                this->Chart_Grid_Search->TabIndex = 1;
                title4->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                                                            static_cast<System::Byte>(0)));
                title4->Name = L"Chart_Title";
                title4->Text = L"Chart Grid Search";
                this->Chart_Grid_Search->Titles->Add(title4);
                this->Chart_Grid_Search->MouseEnter += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Grid_Search_MouseEnter);
                this->Chart_Grid_Search->MouseLeave += gcnew System::EventHandler(this, &Form_Neural_Network::Chart_Grid_Search_MouseLeave);
                // 
                // Button_Chart_Grid_Search_Reset
                // 
                this->Button_Chart_Grid_Search_Reset->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
                this->Button_Chart_Grid_Search_Reset->Location = System::Drawing::Point(592, 3);
                this->Button_Chart_Grid_Search_Reset->Name = L"Button_Chart_Grid_Search_Reset";
                this->Button_Chart_Grid_Search_Reset->Size = System::Drawing::Size(75, 23);
                this->Button_Chart_Grid_Search_Reset->TabIndex = 2;
                this->Button_Chart_Grid_Search_Reset->Text = L"Reset";
                this->Button_Chart_Grid_Search_Reset->UseVisualStyleBackColor = true;
                this->Button_Chart_Grid_Search_Reset->Click += gcnew System::EventHandler(this, &Form_Neural_Network::Button_Chart_Grid_Search_Reset__Click);
                // 
                // Form_Neural_Network
                // 
                this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
                this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
                this->ClientSize = System::Drawing::Size(670, 453);
                this->Controls->Add(this->Panel_Button);
                this->Controls->Add(this->Panel_Information);
                this->Controls->Add(this->Panel_Chart_Loss);
                this->Controls->Add(this->Panel_Chart_Accuracy);
                this->Controls->Add(this->Panel_Chart_Output);
                this->Controls->Add(this->Panel_Chart_Grid_Search);
                this->Name = L"Form_Neural_Network";
                this->Text = L"Neural Network";
                this->Panel_Button->ResumeLayout(false);
                this->Panel_Information->ResumeLayout(false);
                this->Panel_Chart_Loss->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Loss_Y_Minimum))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Loss_Y_Maximum))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Loss))->EndInit();
                this->Panel_Chart_Accuracy->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Accuracy_Y_Minimum))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Accuracy_Y_Maximum))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Accuracy))->EndInit();
                this->Panel_Chart_Output->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Output))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Output_Y_Minimum))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->NumericUpDown_Chart_Output_Y_Maximum))->EndInit();
                this->Panel_Chart_Grid_Search->ResumeLayout(false);
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->DataGridView_Grid_Search))->EndInit();
                (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->Chart_Grid_Search))->EndInit();
                this->ResumeLayout(false);

            }
    #pragma endregion

        private: System::Windows::Forms::DataVisualization::Charting::DataPoint^ Form_Neural_Network::Chart_Series_Find_Maximum_Y(System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received);
        private: System::Windows::Forms::DataVisualization::Charting::DataPoint^ Form_Neural_Network::Chart_Series_Find_Minimum_Y(System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received);
        private: System::Void Form_Neural_Network::Chart_Series_Smooth_Last_DataPoint(double const variance_received,
                                                                                                                                  double const maximum_Y_value_received,
                                                                                                                                  double const minimum_Y_value_received,
                                                                                                                                  System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received);
        private: System::Void Form_Neural_Network::Chart_Series_Smooth_DataPoint(int const index_received,
                                                                                                                          double const variance_received,
                                                                                                                          double const maximum_Y_value_received,
                                                                                                                          double const minimum_Y_value_received,
                                                                                                                          System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received);
        private: System::Void Button_Chart_Loss_Reset__Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Loss_Rescale_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Accuracy_Reset__Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Output_Reset__Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Grid_Search_Reset__Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Grid_Search_Rescale_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Panel_Information_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Panel_Chart_Loss_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Panel_Chart_Accuracy_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Panel_Chart_Output_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Panel_Chart_Grid_Search_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Accuracy_MouseEnter(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Accuracy_MouseLeave(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Output_MouseEnter(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Output_MouseLeave(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Loss_MouseEnter(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Loss_MouseLeave(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Grid_Search_MouseEnter(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Chart_Grid_Search_MouseLeave(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Training_Stop_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Training_Menu_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Output_Type_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Output_Rescale_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Loss_Difference_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Loss_Apply_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Accuracy_Apply_Click(System::Object^  sender, System::EventArgs^  e);
        private: System::Void Button_Chart_Output_Apply_Click(System::Object^  sender, System::EventArgs^  e);
};
    }
}
