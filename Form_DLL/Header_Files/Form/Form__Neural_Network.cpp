#include "stdafx.hpp"

#include <Enums/Enum_Type_Chart.hpp>
#include <Form/Form__Neural_Network.h>
#include <Math/Math.hpp>

#include <iostream> // printf
#include <msclr\marshal_cppstd.h> // System::String^ to std::string

#include <UI/Dialog_Box.hpp>

namespace MyEA
{
    namespace Form
    {
        using namespace System::Windows::Forms::DataVisualization::Charting;
        
        #define INTERVAL_X 10.0
        #define SHIFT_X 4.0

        // Initialize chart.
        delegate void delegate__Form_Neural_Network__Chart_Initialize(unsigned int const type_chart_received, unsigned int const number_series_received);
        System::Void Form_Neural_Network::Chart_Initialize(unsigned int const type_chart_received, unsigned int const number_series_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Initialize^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Initialize(this, &Form_Neural_Network::Chart_Initialize);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  type_chart_received,
                                  number_series_received);

                delete(tmp_ptr_invoke_action);

                return;
            }

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_CHART>(type_chart_received))
            {
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS: this->Chart_Loss_Initialize(number_series_received); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY: this->Chart_Accuracy_Initialize(number_series_received); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT: this->Chart_Output_Initialize(number_series_received); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH: this->Chart_Grid_Search_Initialize(); break;
            }
        }

        delegate void delegate__Form_Neural_Network__Chart_Loss_Initialize(unsigned int const number_series_received);
        System::Void Form_Neural_Network::Chart_Loss_Initialize(unsigned int const number_series_received)
        {
            if(number_series_received < 1u) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Loss_Initialize^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Loss_Initialize(this, &Form_Neural_Network::Chart_Loss_Initialize);
                
                this->Invoke(tmp_ptr_invoke_action, number_series_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            if(this->_ptr_chart_parameters_loss->number_series != 0u) { return; }

            this->_ptr_chart_parameters_loss->total_data_point = 0.0;

            this->_signal_training_stop = false;

            int tmp_color_minimum(125);
            int tmp_color_maximum(255);
            int tmp_color_difference(tmp_color_maximum - tmp_color_minimum);
            int tmp_color_change_step(static_cast<int>(System::Math::Ceiling(static_cast<float>(tmp_color_difference) / static_cast<float>(number_series_received - 1u))));
            
            for(unsigned int i(0u); i != number_series_received; ++i)
            {
                Series^ tmp_ptr_series = gcnew Series;
                tmp_ptr_series->Name = L"SERIES_LOSS_TRAINING_" + i.ToString();
                tmp_ptr_series->Color = Color::FromArgb(tmp_color_maximum - i * tmp_color_change_step,
                                                                             0,
                                                                             0);
                tmp_ptr_series->ChartType = SeriesChartType::FastLine;
                this->_ptr_chart_parameters_loss->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_loss->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_loss->y_maximum_value.Add(-DBL_MAX);
                this->Chart_Loss->Series->Add(tmp_ptr_series);



                tmp_ptr_series = gcnew Series;
                tmp_ptr_series->Name = L"SERIES_LOSS_VALIDATING_" + i.ToString();
                tmp_ptr_series->Color = Color::FromArgb(0,
                                                                             tmp_color_maximum - i * tmp_color_change_step,
                                                                             0);
                tmp_ptr_series->ChartType = SeriesChartType::FastLine;
                this->_ptr_chart_parameters_loss->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_loss->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_loss->y_maximum_value.Add(-DBL_MAX);
                this->Chart_Loss->Series->Add(tmp_ptr_series);



                tmp_ptr_series = gcnew Series;
                tmp_ptr_series->Name = L"SERIES_LOSS_TESTING_" + i.ToString();
                tmp_ptr_series->Color = Color::FromArgb(0,
                                                                             i * tmp_color_change_step,
                                                                             tmp_color_maximum - i * tmp_color_change_step);
                tmp_ptr_series->ChartType = SeriesChartType::FastLine;
                this->_ptr_chart_parameters_loss->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_loss->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_loss->y_maximum_value.Add(-DBL_MAX);
                this->Chart_Loss->Series->Add(tmp_ptr_series);



                tmp_ptr_series = gcnew Series;
                tmp_ptr_series->Name = L"SERIES_LOSS_TESTING_VS_TRAINING_" + i.ToString();
                tmp_ptr_series->Color = Color::FromArgb(tmp_color_maximum - i * tmp_color_change_step,
                                                                             0,
                                                                             tmp_color_maximum - i * tmp_color_change_step);
                tmp_ptr_series->ChartType = SeriesChartType::FastLine;
                this->_ptr_chart_parameters_loss->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_loss->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_loss->y_maximum_value.Add(-DBL_MAX);
                this->Chart_Loss->Series->Add(tmp_ptr_series);



                tmp_ptr_series = gcnew Series;
                tmp_ptr_series->Name = L"SERIES_LOSS_TESTING_VS_VALIDATING_" + i.ToString();
                tmp_ptr_series->Color = Color::FromArgb(tmp_color_maximum - i * tmp_color_change_step,
                                                                             (tmp_color_maximum - i * tmp_color_change_step) / 2,
                                                                             0);
                tmp_ptr_series->ChartType = SeriesChartType::FastLine;
                this->_ptr_chart_parameters_loss->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_loss->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_loss->y_maximum_value.Add(-DBL_MAX);
                this->Chart_Loss->Series->Add(tmp_ptr_series);



                tmp_ptr_series = gcnew Series;
                tmp_ptr_series->Name = L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + i.ToString();
                tmp_ptr_series->Color = Color::FromArgb((tmp_color_maximum - i * tmp_color_change_step) / 2,
                                                                             tmp_color_maximum - i * tmp_color_change_step,
                                                                             (tmp_color_maximum - i * tmp_color_change_step) / 2);
                tmp_ptr_series->ChartType = SeriesChartType::FastLine;
                this->_ptr_chart_parameters_loss->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_loss->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_loss->y_maximum_value.Add(-DBL_MAX);
                this->Chart_Loss->Series->Add(tmp_ptr_series);
            }

            this->_ptr_chart_parameters_loss->number_series = number_series_received;
        }

        delegate void delegate__Form_Neural_Network__Chart_Accuracy_Initialize(unsigned int const number_series_received);
        System::Void Form_Neural_Network::Chart_Accuracy_Initialize(unsigned int const number_series_received)
        {
            if(number_series_received < 1u) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Accuracy_Initialize^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Accuracy_Initialize(this, &Form_Neural_Network::Chart_Accuracy_Initialize);
                
                this->Invoke(tmp_ptr_invoke_action, number_series_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            if(this->_ptr_chart_parameters_accuracy->number_series != 0u) { return; }

            this->_ptr_chart_parameters_accuracy->total_data_point = 0.0;

            this->_signal_training_stop = false;

            int tmp_color_minimum(125);
            int tmp_color_maximum(255);
            int tmp_color_difference(tmp_color_maximum - tmp_color_minimum);
            int tmp_color_change_step(static_cast<int>(System::Math::Ceiling(static_cast<float>(tmp_color_difference) / static_cast<float>(number_series_received - 1u))));

            for(unsigned int i(0u); i != number_series_received; ++i)
            {
                Series^ tmp_ptr_series_training = gcnew Series;
                tmp_ptr_series_training->Name = L"SERIES_ACCURACY_TRAINING_" + i.ToString();
                tmp_ptr_series_training->Color = Color::FromArgb(tmp_color_maximum - i * tmp_color_change_step,
                                                                                        0,
                                                                                        0);
                tmp_ptr_series_training->ChartType = SeriesChartType::FastLine;

                this->Chart_Accuracy->Series->Add(tmp_ptr_series_training);
                this->_ptr_chart_parameters_accuracy->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_accuracy->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_accuracy->y_maximum_value.Add(-DBL_MAX);

                Series^ tmp_ptr_series_validating = gcnew Series;
                tmp_ptr_series_validating->Name = L"SERIES_ACCURACY_VALIDATING_" + i.ToString();
                tmp_ptr_series_validating->Color = Color::FromArgb(0,
                                                                                            tmp_color_maximum - i * tmp_color_change_step,
                                                                                            0);
                tmp_ptr_series_validating->ChartType = SeriesChartType::FastLine;

                this->Chart_Accuracy->Series->Add(tmp_ptr_series_validating);
                this->_ptr_chart_parameters_accuracy->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_accuracy->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_accuracy->y_maximum_value.Add(-DBL_MAX);

                Series^ tmp_ptr_series_testing = gcnew Series;
                tmp_ptr_series_testing->Name = L"SERIES_ACCURACY_TESTING_" + i.ToString();
                tmp_ptr_series_testing->Color = Color::FromArgb(0,
                                                                                        i * tmp_color_change_step,
                                                                                        tmp_color_maximum - i * tmp_color_change_step);
                tmp_ptr_series_testing->ChartType = SeriesChartType::FastLine;

                this->Chart_Accuracy->Series->Add(tmp_ptr_series_testing);
                this->_ptr_chart_parameters_accuracy->y_current_value.Add(0.0);
                this->_ptr_chart_parameters_accuracy->y_minimum_value.Add(DBL_MAX);
                this->_ptr_chart_parameters_accuracy->y_maximum_value.Add(-DBL_MAX);
            }

            this->_ptr_chart_parameters_accuracy->number_series = number_series_received;
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Output_Initialize(unsigned int const number_series_received);
        System::Void Form_Neural_Network::Chart_Output_Initialize(unsigned int const number_series_received)
        {
            if(number_series_received < 1u) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Output_Initialize^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Output_Initialize(this, &Form_Neural_Network::Chart_Output_Initialize);
                
                this->Invoke(tmp_ptr_invoke_action, number_series_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            if(this->_ptr_chart_parameters_output->number_series != 0u) { return; }

            this->_ptr_chart_parameters_output->total_data_point = 0.0;

            int tmp_color_minimum(125);
            int tmp_color_maximum(255);
            int tmp_color_difference(tmp_color_maximum - tmp_color_minimum);
            int tmp_color_change_step(static_cast<int>(System::Math::Ceiling(static_cast<float>(tmp_color_difference) / static_cast<float>(number_series_received - 1u))));

            for(unsigned int i(0u); i != number_series_received; ++i)
            {
                Series^ tmp_ptr_series_training = gcnew Series;
                tmp_ptr_series_training->Name = L"SERIES_OUTPUT_TRAINING_" + i.ToString();
                tmp_ptr_series_training->Color = Color::FromArgb(tmp_color_maximum - i * tmp_color_change_step,
                                                                                        0,
                                                                                        0);
                tmp_ptr_series_training->ChartType = SeriesChartType::FastPoint;
                this->Chart_Output->Series->Add(tmp_ptr_series_training);

                Series^ tmp_ptr_series_validating = gcnew Series;
                tmp_ptr_series_validating->Name = L"SERIES_OUTPUT_VALIDATING_" + i.ToString();
                tmp_ptr_series_validating->Color = Color::FromArgb(0,
                                                                                            tmp_color_maximum - i * tmp_color_change_step,
                                                                                            0);
                tmp_ptr_series_validating->ChartType = SeriesChartType::FastPoint;
                this->Chart_Output->Series->Add(tmp_ptr_series_validating);

                Series^ tmp_ptr_series_testing = gcnew Series;
                tmp_ptr_series_testing->Name = L"SERIES_OUTPUT_TESTING_" + i.ToString();
                tmp_ptr_series_testing->Color = Color::FromArgb(0,
                                                                                        i * tmp_color_change_step,
                                                                                        tmp_color_maximum - i * tmp_color_change_step);
                tmp_ptr_series_testing->ChartType = SeriesChartType::FastPoint;
                this->Chart_Output->Series->Add(tmp_ptr_series_testing);
            }

            this->_ptr_chart_parameters_output->y_minimum_value.Add((std::numeric_limits<double>::max)());
            this->_ptr_chart_parameters_output->y_maximum_value.Add(-(std::numeric_limits<double>::max)());
            this->_ptr_chart_parameters_output->number_series = number_series_received;
        }

        delegate void delegate__Form_Neural_Network__Chart_Grid_Search_Initialize(void);
        System::Void Form_Neural_Network::Chart_Grid_Search_Initialize(void)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Grid_Search_Initialize^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Grid_Search_Initialize(this, &Form_Neural_Network::Chart_Grid_Search_Initialize);
                
                this->Invoke(tmp_ptr_invoke_action);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            if(this->_ptr_chart_parameters_grid_search->number_series != 0u) { return; }
            
            this->_ptr_chart_parameters_grid_search->total_data_point = 0.0;

            /*
            Series^ tmp_ptr_series = gcnew Series;
            tmp_ptr_series->Name = L"SERIES_GRID_SEARCH_TRAINING";
            tmp_ptr_series->MarkerColor = System::Drawing::Color::Red;
            tmp_ptr_series->MarkerStyle = System::Windows::Forms::DataVisualization::Charting::MarkerStyle::Circle;
            tmp_ptr_series->ChartType = SeriesChartType::FastPoint;
            this->Chart_Grid_Search->Series->Add(tmp_ptr_series);

            tmp_ptr_series = gcnew Series;
            tmp_ptr_series->Name = L"SERIES_GRID_SEARCH_VALIDATING";
            tmp_ptr_series->MarkerColor = System::Drawing::Color::Green;
            tmp_ptr_series->MarkerStyle = System::Windows::Forms::DataVisualization::Charting::MarkerStyle::Circle;
            tmp_ptr_series->ChartType = SeriesChartType::FastPoint;
            this->Chart_Grid_Search->Series->Add(tmp_ptr_series);

            tmp_ptr_series = gcnew Series;
            tmp_ptr_series->Name = L"SERIES_GRID_SEARCH_TESTING";
            tmp_ptr_series->MarkerColor = System::Drawing::Color::Blue;
            tmp_ptr_series->MarkerStyle = System::Windows::Forms::DataVisualization::Charting::MarkerStyle::Circle;
            tmp_ptr_series->ChartType = SeriesChartType::FastPoint;
            this->Chart_Grid_Search->Series->Add(tmp_ptr_series);
            */

            Series^ tmp_ptr_series = gcnew Series;
            tmp_ptr_series->Name = L"SERIES_GRID_SEARCH_TRAINING";
            tmp_ptr_series->MarkerColor = System::Drawing::Color::Red;
            tmp_ptr_series->ChartType = SeriesChartType::Area;
            this->Chart_Grid_Search->Series->Add(tmp_ptr_series);

            tmp_ptr_series = gcnew Series;
            tmp_ptr_series->Name = L"SERIES_GRID_SEARCH_VALIDATING";
            tmp_ptr_series->MarkerColor = System::Drawing::Color::Green;
            tmp_ptr_series->ChartType = SeriesChartType::Area;
            this->Chart_Grid_Search->Series->Add(tmp_ptr_series);

            tmp_ptr_series = gcnew Series;
            tmp_ptr_series->Name = L"SERIES_GRID_SEARCH_TESTING";
            tmp_ptr_series->MarkerColor = System::Drawing::Color::Blue;
            tmp_ptr_series->ChartType = SeriesChartType::Area;
            this->Chart_Grid_Search->Series->Add(tmp_ptr_series);

            this->_ptr_chart_parameters_grid_search->number_series = 1u;
        }
        // |END| Initialize chart. |END|
        
        delegate void delegate__Form_Neural_Network__Chart_Use_Datapoint_Training(bool const use_datapoint_training_received);
        System::Void Form_Neural_Network::Chart_Use_Datapoint_Training(bool const use_datapoint_training_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Use_Datapoint_Training^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Use_Datapoint_Training(this, &Form_Neural_Network::Chart_Use_Datapoint_Training);
                
                this->Invoke(tmp_ptr_invoke_action, use_datapoint_training_received);

                delete(tmp_ptr_invoke_action);

                return;
            }
            
            this->_use_Datapoint_Training = use_datapoint_training_received;
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Use_Datapoint_Difference(bool const use_datapoint_training_received);
        System::Void Form_Neural_Network::Chart_Use_Datapoint_Difference(bool const use_datapoint_difference_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Use_Datapoint_Difference^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Use_Datapoint_Difference(this, &Form_Neural_Network::Chart_Use_Datapoint_Difference);
                
                this->Invoke(tmp_ptr_invoke_action, use_datapoint_difference_received);

                delete(tmp_ptr_invoke_action);

                return;
            }
            
            this->_use_Datapoint_Difference = use_datapoint_difference_received;
        }

        delegate void delegate__Form_Neural_Network__Chart_Total_Means(unsigned int const total_means_received);
        System::Void Form_Neural_Network::Chart_Total_Means(unsigned int const total_means_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Total_Means^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Total_Means(this, &Form_Neural_Network::Chart_Total_Means);
                
                this->Invoke(tmp_ptr_invoke_action, total_means_received);

                delete(tmp_ptr_invoke_action);

                return;
            }
            
            this->Label_Loss_Mean->Text = "Mean " + total_means_received.ToString();
            this->_ptr_chart_parameters_loss->struct_training_mean.Set__Total_Means(total_means_received);
            this->_ptr_chart_parameters_loss->struct_validating_mean.Set__Total_Means(total_means_received);
            this->_ptr_chart_parameters_loss->struct_testing_mean.Set__Total_Means(total_means_received);

            this->Label_Accuracy_Mean->Text = "Mean " + total_means_received.ToString();
            this->_ptr_chart_parameters_accuracy->struct_training_mean.Set__Total_Means(total_means_received);
            this->_ptr_chart_parameters_accuracy->struct_validating_mean.Set__Total_Means(total_means_received);
            this->_ptr_chart_parameters_accuracy->struct_testing_mean.Set__Total_Means(total_means_received);
        }

        // Interval.
        delegate void delegate__Form_Neural_Network__Chart_Reset(unsigned int const type_chart_received);
        System::Void Form_Neural_Network::Chart_Reset(unsigned int const type_chart_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Reset^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Reset(this, &Form_Neural_Network::Chart_Reset);
                
                this->Invoke(tmp_ptr_invoke_action, type_chart_received);

                delete(tmp_ptr_invoke_action);

                return;
            }

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_CHART>(type_chart_received))
            {
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS: this->Button_Chart_Loss_Reset__Click(nullptr, nullptr); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY: this->Button_Chart_Accuracy_Reset__Click(nullptr, nullptr); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT: this->Button_Chart_Output_Reset__Click(nullptr, nullptr); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH: this->Button_Chart_Grid_Search_Reset__Click(nullptr, nullptr); break;
            }
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Rescale(unsigned int const type_chart_received);
        System::Void Form_Neural_Network::Chart_Rescale(unsigned int const type_chart_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Rescale^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Rescale(this, &Form_Neural_Network::Chart_Rescale);
                
                this->Invoke(tmp_ptr_invoke_action, type_chart_received);

                delete(tmp_ptr_invoke_action);

                return;
            }

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_CHART>(type_chart_received))
            {
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS: this->Button_Chart_Loss_Rescale_Click(nullptr, nullptr); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT: this->Button_Chart_Output_Rescale_Click(nullptr, nullptr); break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH: this->Button_Chart_Grid_Search_Rescale_Click(nullptr, nullptr); break;
            }
        }
        // |END| Interval. |END|
        
        // Add point.
        delegate void delegate__Form_Neural_Network__Chart_Add_Point(unsigned int const type_chart_received,
                                                                                                        unsigned int const index_series_received,
                                                                                                        unsigned int const type_received,
                                                                                                        double const x_received,
                                                                                                        double const y_received);
        System::Void Form_Neural_Network::Chart_Add_Point(unsigned int const type_chart_received,
                                                                                        unsigned int const index_series_received,
                                                                                        unsigned int const type_received,
                                                                                        double const x_received,
                                                                                        double const y_received)
        {
            if(type_chart_received >= MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LENGTH) { return; }
            else if(x_received < 0.0) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Add_Point^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Add_Point(this, &Form_Neural_Network::Chart_Add_Point);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  type_chart_received,
                                  index_series_received,
                                  type_received,
                                  x_received,
                                  y_received);

                delete(tmp_ptr_invoke_action);

                return;
            }

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_CHART>(type_chart_received))
            {
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS:
                    this->Chart_Loss_Add_Point(index_series_received,
                                                              type_received,
                                                              x_received,
                                                              y_received);
                            break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY:
                    this->Chart_Accuracy_Add_Point(index_series_received,
                                                                      type_received,
                                                                      x_received,
                                                                      y_received);
                            break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT:
                    this->Chart_Output_Add_Point(index_series_received,
                                                                 type_received,
                                                                 x_received,
                                                                 y_received);
                            break;
                case MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH:
                    this->Chart_Grid_Search_Add_Point(type_received,
                                                                          x_received,
                                                                          y_received);
                            break;
            }
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Loss_Diff(unsigned int const index_series_received,
                                                                                                       unsigned int const type_received,
                                                                                                       double const x_received);
        System::Void Form_Neural_Network::Chart_Loss_Diff(unsigned int const index_series_received,
                                                                                      unsigned int const type_received,
                                                                                      double const x_received)
        {
            if(type_received >= MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH || type_received == MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING) { return; }
            else if(x_received < 0.0) { return; }
            else if(this->_use_Datapoint_Difference == false) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Loss_Diff^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Loss_Diff(this, &Form_Neural_Network::Chart_Loss_Diff);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  index_series_received,
                                  type_received,
                                  x_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }

            if(index_series_received >= this->_ptr_chart_parameters_loss->number_series) { return; }
            
            unsigned int const tmp_index_series_shift(index_series_received * 6u);

            double tmp_y_diff;
            
            switch(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_received))
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                    // Validating vs Training
                    tmp_y_diff = MyEA::Math::Absolute<double>(this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 1u] - this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift]);

                    this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 5u] = tmp_y_diff;

                    if(tmp_y_diff < this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 5u])
                    { this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 5u] = tmp_y_diff; }
                        
                    if(tmp_y_diff > this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 5u])
                    { this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 5u] = tmp_y_diff; }
                        
                    if(this->_use_Datapoint_Training || index_series_received != 0u)
                    {
                        this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + index_series_received.ToString()]->Points->AddXY(x_received, tmp_y_diff);
                        
                        this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_loss->precision,
                                                                                         this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 5u],
                                                                                         this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 5u],
                                                                                         this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + index_series_received.ToString()]->Points);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                    // Testing vs Training.
                    tmp_y_diff = MyEA::Math::Absolute<double>(this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 2u] - this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift]);

                    this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 3u] = tmp_y_diff;

                    if(tmp_y_diff < this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 3u])
                    { this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 3u] = tmp_y_diff; }

                    if(tmp_y_diff > this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 3u])
                    { this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 3u] = tmp_y_diff; }

                    if(this->_use_Datapoint_Training || index_series_received != 0u)
                    {
                        this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_TRAINING_" + index_series_received.ToString()]->Points->AddXY(x_received, tmp_y_diff);
                        
                        this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_loss->precision,
                                                                                         this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 3u],
                                                                                         this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 3u],
                                                                                         this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_TRAINING_" + index_series_received.ToString()]->Points);
                    }

                    // Testing vs Validating
                    tmp_y_diff = MyEA::Math::Absolute<double>(this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 2u] - this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 1u]);

                    this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 4u] = tmp_y_diff;

                    if(tmp_y_diff < this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 4u])
                    { this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 4u] = tmp_y_diff; }
                        
                    if(tmp_y_diff > this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 4u])
                    { this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 4u] = tmp_y_diff; }
                        
                    if(this->_use_Datapoint_Training || index_series_received != 0u)
                    {
                        this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_VALIDATING_" + index_series_received.ToString()]->Points->AddXY(x_received, tmp_y_diff);
                        
                        this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_loss->precision,
                                                                                         this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 4u],
                                                                                         this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 4u],
                                                                                         this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_VALIDATING_" + index_series_received.ToString()]->Points);
                    }
                        break;
                default: break;
            }
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Loss_Add_Point(unsigned int const index_series_received,
                                                                                                                unsigned int const type_loss_received,
                                                                                                                double const x_received,
                                                                                                                double y_received);
        System::Void Form_Neural_Network::Chart_Loss_Add_Point(unsigned int const index_series_received,
                                                                                                 unsigned int const type_loss_received,
                                                                                                 double const x_received,
                                                                                                 double y_received)
        {
            if(type_loss_received >= MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH) { return; }
            else if(x_received < 0.0) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Loss_Add_Point^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Loss_Add_Point(this, &Form_Neural_Network::Chart_Loss_Add_Point);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  index_series_received,
                                  type_loss_received,
                                  x_received,
                                  y_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }

            if(index_series_received >= this->_ptr_chart_parameters_loss->number_series) { return; }
            
            if(this->_ptr_chart_parameters_loss->total_data_point == 0.0) { this->Chart_Loss->ChartAreas[0u]->AxisX->Minimum = x_received; }

            unsigned int const tmp_index_series_shift(index_series_received * 6u);

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_loss_received))
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                        if(index_series_received == 0u) { this->_ptr_chart_parameters_loss->struct_training_mean.Add(y_received); }

                        this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift] = y_received;

                        if(y_received < this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift])
                        { this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift] = y_received; }
                        
                        if(y_received > this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift])
                        { this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift] = y_received; }

                        if(this->_use_Datapoint_Training || index_series_received != 0u)
                        {
                            this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received);
                        
                            this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_loss->precision,
                                                                                             this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift],
                                                                                             this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift],
                                                                                             this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_" + index_series_received.ToString()]->Points);
                        }
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                        if(index_series_received == 0u) { this->_ptr_chart_parameters_loss->struct_validating_mean.Add(y_received); }

                        this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 1u] = y_received;

                        if(y_received < this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 1u])
                        { this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 1u] = y_received; }
                        
                        if(y_received > this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 1u])
                        { this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 1u] = y_received; }
                        
                        if(this->_use_Datapoint_Training || index_series_received != 0u)
                        {
                            this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received);
                       
                            this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_loss->precision,
                                                                                             this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 1u],
                                                                                             this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 1u],
                                                                                             this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_" + index_series_received.ToString()]->Points);
                        }
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                        if(index_series_received == 0u) { this->_ptr_chart_parameters_loss->struct_testing_mean.Add(y_received); }

                        this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 2u] = y_received;

                        if(y_received < this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 2u])
                        { this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 2u] = y_received; }
                        
                        if(y_received > this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 2u])
                        { this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 2u] = y_received; }
                        
                        if(this->_use_Datapoint_Training || index_series_received != 0u)
                        {
                            this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received);
                        
                            this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_loss->precision,
                                                                                             this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 2u],
                                                                                             this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 2u],
                                                                                             this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_" + index_series_received.ToString()]->Points);
                        }
                            break;
                default: break;
            }

            this->_ptr_chart_parameters_loss->total_data_point = x_received;
            
            // Shift view by 4%.
            double const tmp_X_size(x_received - this->Chart_Loss->ChartAreas[0u]->AxisX->Minimum),
                                tmp_X_shift(x_received + ceil(tmp_X_size * SHIFT_X / 100.0)),
                                tmp_X_scale(ceil(tmp_X_shift / INTERVAL_X) * INTERVAL_X);
            if(this->Chart_Loss->ChartAreas[0u]->AxisX->Maximum < tmp_X_scale || this->Chart_Loss->ChartAreas[0u]->AxisX->Maximum == NAN)
            {
                this->Chart_Loss->ChartAreas[0u]->AxisX->Maximum = tmp_X_scale + 1.0;

                this->ScaleView_Interval(this->Chart_Loss->ChartAreas[0u]);

                this->Compute_Precision(this->_ptr_chart_parameters_loss, this->Chart_Loss->ChartAreas[0u]);

                this->ScaleView_Zoom(tmp_X_scale + 1.0,
                                                  this->_ptr_chart_parameters_loss,
                                                  this->Chart_Loss->ChartAreas[0u]);
            }

            if(this->_use_Datapoint_Training || index_series_received != 0u)
            {
                this->RichTextBox_DataPoint_Loss->Text = "DataPoint loss: " + (this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_0"]->Points->Count
                                                                                                           + this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_0"]->Points->Count
                                                                                                           + this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_0"]->Points->Count).ToString();
            }

            this->Output_Loss_Information(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_loss_received), 0);
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Accuracy_Add_Point(unsigned int const index_series_received,
                                                                                                                        unsigned int const type_accuracy_received,
                                                                                                                        double const x_received,
                                                                                                                        double y_received);
        System::Void Form_Neural_Network::Chart_Accuracy_Add_Point(unsigned int const index_series_received,
                                                                                                        unsigned int const type_accuracy_received,
                                                                                                        double const x_received,
                                                                                                        double y_received)
        {
            if(type_accuracy_received >= MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH) { return; }
            else if(y_received < 0.0 || x_received < 0.0) { return; }
            else if(y_received > 100.0) { y_received = 100.0; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Accuracy_Add_Point^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Accuracy_Add_Point(this, &Form_Neural_Network::Chart_Accuracy_Add_Point);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  index_series_received,
                                  type_accuracy_received,
                                  x_received,
                                  y_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }

            if(index_series_received >= this->_ptr_chart_parameters_accuracy->number_series) { return; }
            
            if(this->_ptr_chart_parameters_accuracy->total_data_point == 0.0) { this->Chart_Accuracy->ChartAreas[0u]->AxisX->Minimum = x_received; }

            unsigned int const tmp_index_series_shift(index_series_received * 3u);

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_accuracy_received))
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                        if(index_series_received == 0u) { this->_ptr_chart_parameters_accuracy->struct_training_mean.Add(y_received); }

                        this->_ptr_chart_parameters_accuracy->y_current_value[tmp_index_series_shift] = y_received;

                        if(y_received < this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift])
                        { this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift] = y_received; }
                        
                        if(y_received > this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift])
                        { this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift] = y_received; }
                        
                        if(this->_use_Datapoint_Training || index_series_received != 0u)
                        {
                            this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TRAINING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received);
                        
                            this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_accuracy->precision,
                                                                                             this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift],
                                                                                             this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift],
                                                                                             this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TRAINING_" + index_series_received.ToString()]->Points);
                        }
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                        if(index_series_received == 0u) { this->_ptr_chart_parameters_accuracy->struct_validating_mean.Add(y_received); }

                        this->_ptr_chart_parameters_accuracy->y_current_value[tmp_index_series_shift + 1u] = y_received;

                        if(y_received < this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 1u])
                        { this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 1u] = y_received; }
                        
                        if(y_received > this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 1u])
                        { this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 1u] = y_received; }
                        
                        if(this->_use_Datapoint_Training || index_series_received != 0u)
                        {
                            this->Chart_Accuracy->Series[L"SERIES_ACCURACY_VALIDATING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received);
                            
                            this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_accuracy->precision,
                                                                                             this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 1u],
                                                                                             this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 1u],
                                                                                             this->Chart_Accuracy->Series[L"SERIES_ACCURACY_VALIDATING_" + index_series_received.ToString()]->Points);
                        }
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                        if(index_series_received == 0u) { this->_ptr_chart_parameters_accuracy->struct_testing_mean.Add(y_received); }

                        this->_ptr_chart_parameters_accuracy->y_current_value[tmp_index_series_shift + 2u] = y_received;

                        if(y_received < this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 2u])
                        { this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 2u] = y_received; }
                        
                        if(y_received > this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 2u])
                        { this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 2u] = y_received; }
                        
                        if(this->_use_Datapoint_Training || index_series_received != 0u)
                        {
                            this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TESTING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received);
                            
                            this->Chart_Series_Smooth_Last_DataPoint(this->_ptr_chart_parameters_accuracy->precision,
                                                                                             this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 2u],
                                                                                             this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 2u],
                                                                                             this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TESTING_" + index_series_received.ToString()]->Points);
                        }
                            break;
            }
            
            // Shift view by 4%.
            double const tmp_X_size(x_received - this->Chart_Accuracy->ChartAreas[0u]->AxisX->Minimum),
                                tmp_X_shift(x_received + ceil(tmp_X_size * SHIFT_X / 100.0)),
                                tmp_X_scale(ceil(tmp_X_shift / INTERVAL_X) * INTERVAL_X);
            if(this->Chart_Accuracy->ChartAreas[0u]->AxisX->Maximum < tmp_X_scale || this->Chart_Accuracy->ChartAreas[0u]->AxisX->Maximum == NAN)
            {
                this->Chart_Accuracy->ChartAreas[0u]->AxisX->Maximum = tmp_X_scale + 1.0;
                
                this->ScaleView_Interval(this->Chart_Accuracy->ChartAreas[0u]);

                this->Compute_Precision(this->_ptr_chart_parameters_accuracy, this->Chart_Accuracy->ChartAreas[0u]);
                
                this->ScaleView_Zoom(tmp_X_scale + 1.0,
                                                  this->_ptr_chart_parameters_accuracy,
                                                  this->Chart_Accuracy->ChartAreas[0u]);
            }

            this->_ptr_chart_parameters_accuracy->total_data_point = x_received;
            
            if(this->_use_Datapoint_Training || index_series_received != 0u)
            {
                this->RichTextBox_DataPoint_Accuracy->Text = "DataPoint accuracy: " + (this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TRAINING_0"]->Points->Count
                                                                                                                         + this->Chart_Accuracy->Series[L"SERIES_ACCURACY_VALIDATING_0"]->Points->Count
                                                                                                                         + this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TESTING_0"]->Points->Count).ToString();
            }

            this->Output_Accuracy_Information(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_accuracy_received), 0);
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Output_Add_Point(unsigned int const index_series_received,
                                                                                                                    unsigned int const type_accuracy_received,
                                                                                                                    double const x_received,
                                                                                                                    double y_received);
        System::Void Form_Neural_Network::Chart_Output_Add_Point(unsigned int const index_series_received,
                                                                                                   unsigned int const type_accuracy_received,
                                                                                                   double const x_received,
                                                                                                   double y_received)
        {
            if(type_accuracy_received >= MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Output_Add_Point^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Output_Add_Point(this, &Form_Neural_Network::Chart_Output_Add_Point);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  index_series_received,
                                  type_accuracy_received,
                                  x_received,
                                  y_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }

            if(index_series_received >= this->_ptr_chart_parameters_output->number_series) { return; }
            
            if(this->_ptr_chart_parameters_output->total_data_point == 0.0) { this->Chart_Output->ChartAreas[0u]->AxisX->Minimum = x_received; }

            double tmp_distance;

            if(this->_ptr_chart_parameters_output->total_data_point == 0.0)
            {
                this->_ptr_chart_parameters_output->y_maximum_value[0] = y_received;

                this->_ptr_chart_parameters_output->y_minimum_value[0] = y_received;
                
                tmp_distance = 0.1;
                
                this->Chart_Output->ChartAreas[0]->AxisY->Maximum = y_received + tmp_distance;

                this->Chart_Output->ChartAreas[0]->AxisY->Minimum = y_received - tmp_distance;
            }
            else
            {
                if(this->_ptr_chart_parameters_output->y_maximum_value[0] < y_received)
                {
                    this->_ptr_chart_parameters_output->y_maximum_value[0] = y_received;
                    
                    tmp_distance = MyEA::Math::Maximum<double>((y_received - this->_ptr_chart_parameters_output->y_minimum_value[0]) * 10.0 / 100.0, 0.1);
                    
                    this->Chart_Output->ChartAreas[0]->AxisY->Maximum = y_received + tmp_distance;

                    this->Chart_Output->ChartAreas[0]->AxisY->Minimum = this->_ptr_chart_parameters_output->y_minimum_value[0] - tmp_distance;
                }

                if(this->_ptr_chart_parameters_output->y_minimum_value[0] > y_received)
                {
                    this->_ptr_chart_parameters_output->y_minimum_value[0] = y_received;
                    
                    tmp_distance = MyEA::Math::Maximum<double>((this->_ptr_chart_parameters_output->y_maximum_value[0] - y_received) * 10.0 / 100.0, 0.1);
                    
                    this->Chart_Output->ChartAreas[0]->AxisY->Maximum = this->_ptr_chart_parameters_output->y_maximum_value[0] + tmp_distance;

                    this->Chart_Output->ChartAreas[0]->AxisY->Minimum = y_received - tmp_distance;
                }
            }

            double tmp_shifted_index(0.0);
            
            switch(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_accuracy_received))
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + index_series_received.ToString()]->Points->Count > x_received)
                    {
                        this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + index_series_received.ToString()]->Points[x_received]->YValues[0u] = y_received;

                        this->Chart_Output->Refresh();
                    }
                    else { this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received); }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + index_series_received.ToString()]->Points->Count != 0)
                    { tmp_shifted_index = this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + index_series_received.ToString()]->Points[0]->XValue; }

                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + index_series_received.ToString()]->Points->Count > x_received - tmp_shifted_index)
                    {
                        this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + index_series_received.ToString()]->Points[x_received - tmp_shifted_index]->YValues[0u] = y_received;

                        this->Chart_Output->Refresh();
                    }
                    else { this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received); }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + index_series_received.ToString()]->Points->Count != 0)
                    { tmp_shifted_index = this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + index_series_received.ToString()]->Points[0]->XValue; }

                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + index_series_received.ToString()]->Points->Count > x_received - tmp_shifted_index)
                    {
                        this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + index_series_received.ToString()]->Points[x_received - tmp_shifted_index]->YValues[0u] = y_received;

                        this->Chart_Output->Refresh();
                    }
                    else { this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + index_series_received.ToString()]->Points->AddXY(x_received, y_received); }
                        break;
            }
            
            if(this->_ptr_chart_parameters_output->total_data_point < x_received + 1.0) { this->_ptr_chart_parameters_output->total_data_point = x_received + 1.0; }
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Grid_Search_Add_Point(unsigned int const type_loss_received,
                                                                                                                            double const x_received,
                                                                                                                            double y_received);
        System::Void Form_Neural_Network::Chart_Grid_Search_Add_Point(unsigned int const type_loss_received,
                                                                                                            double const x_received,
                                                                                                            double y_received)
        {
            if(type_loss_received >= MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH) { return; }
            else if(x_received < 0.0) { return; }

            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Grid_Search_Add_Point^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Grid_Search_Add_Point(this, &Form_Neural_Network::Chart_Grid_Search_Add_Point);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  type_loss_received,
                                  x_received,
                                  y_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            if(this->_ptr_chart_parameters_grid_search->total_data_point == 0.0) { this->Chart_Grid_Search->ChartAreas[0u]->AxisX->Minimum = x_received; }

            double tmp_maximum;

            switch(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_loss_received))
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                    if(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points->Count > x_received)
                    {
                        this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points[x_received]->YValues[0u] = y_received;
                        
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points)->YValues[0u];
                        
                        if(tmp_maximum != this->Chart_Grid_Search->ChartAreas[0u]->AxisY->Maximum) { this->Chart_Grid_Search->ChartAreas[0u]->AxisY->Maximum = tmp_maximum; }

                        this->Chart_Grid_Search->Refresh();
                    }
                    else { this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points->AddXY(x_received, y_received); }
                    
                    //this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points->AddXY(x_received, y_received);
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                    if(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points->Count > x_received)
                    {
                        this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points[x_received]->YValues[0u] = y_received;
                        
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points)->YValues[0u];
                        
                        if(tmp_maximum != this->Chart_Grid_Search->ChartAreas[0u]->AxisY->Maximum) { this->Chart_Grid_Search->ChartAreas[0u]->AxisY->Maximum = tmp_maximum; }

                        this->Chart_Grid_Search->Refresh();
                    }
                    else { this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points->AddXY(x_received, y_received); }
                    
                    //this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points->AddXY(x_received, y_received);
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                    if(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points->Count > x_received)
                    {
                        this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points[x_received]->YValues[0u] = y_received;
                        
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points)->YValues[0u];
                        
                        if(tmp_maximum != this->Chart_Grid_Search->ChartAreas[0u]->AxisY->Maximum) { this->Chart_Grid_Search->ChartAreas[0u]->AxisY->Maximum = tmp_maximum; }

                        this->Chart_Grid_Search->Refresh();
                    }
                    else { this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points->AddXY(x_received, y_received); }
                    
                    //this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points->AddXY(x_received, y_received);
                            break;
            }
            
            if(this->_ptr_chart_parameters_grid_search->total_data_point < x_received + 1.0)
            {
                this->Chart_Grid_Search->ChartAreas[0u]->AxisX->Maximum = x_received;

                this->_ptr_chart_parameters_grid_search->total_data_point = x_received + 1.0;
            }
        }
        
        System::Void Form_Neural_Network::Chart_Series_Smooth_Last_DataPoint(double const variance_received,
                                                                                                                       double const maximum_Y_value_received,
                                                                                                                       double const minimum_Y_value_received,
                                                                                                                       System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received)
        {
            this->Chart_Series_Smooth_DataPoint(series_received->Count - 2,
                                                                     variance_received,
                                                                     maximum_Y_value_received,
                                                                     minimum_Y_value_received,
                                                                     series_received);
        }
        
        System::Void Form_Neural_Network::Chart_Series_Smooth_DataPoint(int const index_received,
                                                                                                                double const variance_received,
                                                                                                                double const maximum_Y_value_received,
                                                                                                                double const minimum_Y_value_received,
                                                                                                                System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received)
        {
            if(index_received - 1 >= 0 && index_received + 1 < series_received->Count)
            {
                double const tmp_previous_Y(series_received[index_received - 1]->YValues[0]),
                                    tmp_current_Y(series_received[index_received]->YValues[0]),
                                    tmp_next_Y(series_received[index_received + 1]->YValues[0]);

                if(tmp_previous_Y == tmp_current_Y && tmp_current_Y == tmp_next_Y)
                { series_received->RemoveAt(index_received); }
                else if(tmp_current_Y != maximum_Y_value_received
                          &&
                          tmp_current_Y != minimum_Y_value_received
                          &&
                          MyEA::Math::Absolute<double>((tmp_previous_Y + tmp_next_Y) / 2.0 - tmp_current_Y) <= variance_received)
                { series_received->RemoveAt(index_received); }
            }
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Grid_Search_Add_Column(System::String ^value_received);
        System::Void Form_Neural_Network::Chart_Grid_Search_Add_Column(System::String ^value_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Grid_Search_Add_Column^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Grid_Search_Add_Column(this, &Form_Neural_Network::Chart_Grid_Search_Add_Column);
                
                this->Invoke(tmp_ptr_invoke_action, value_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            this->DataGridView_Grid_Search->Columns->Add(value_received, value_received);
        }
        
        delegate void delegate__Form_Neural_Network__Chart_Grid_Search_Add_Row(unsigned int const index_received, System::String ^value_received);
        System::Void Form_Neural_Network::Chart_Grid_Search_Add_Row(unsigned int const index_received, System::String ^value_received)
        {
            if(this->InvokeRequired)
            {
                delegate__Form_Neural_Network__Chart_Grid_Search_Add_Row^ tmp_ptr_invoke_action = gcnew delegate__Form_Neural_Network__Chart_Grid_Search_Add_Row(this, &Form_Neural_Network::Chart_Grid_Search_Add_Row);
                
                this->Invoke(tmp_ptr_invoke_action,
                                  index_received,
                                  value_received);
                
                delete(tmp_ptr_invoke_action);

                return;
            }
            
            if(index_received == 0u) { this->DataGridView_Grid_Search->Rows->Add(); }

            this->DataGridView_Grid_Search->Rows[this->DataGridView_Grid_Search->RowCount - 1]->Cells[index_received]->Value = value_received;
        }
        // |END| Add point. |END|
        
        System::Void Form_Neural_Network::Output_Loss_Information(enum MyEA::Common::ENUM_TYPE_DATASET const type_loss_received, unsigned int const index_series_received)
        {
            unsigned int const tmp_index_series_shift(index_series_received * 6u);

            switch(type_loss_received)
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                    this->RichTextBox_Loss_Training_Current->Text = this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift].ToString("###0.000000000");
                    this->RichTextBox_Loss_Training_Higher->Text = this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift].ToString("###0.000000000");
                    this->RichTextBox_Loss_Training_Lower->Text = this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift].ToString("###0.000000000");
                    this->RichTextBox_Loss_Training_Mean->Text = this->_ptr_chart_parameters_loss->struct_training_mean.Get__Current_Mean().ToString("###0.000000000");
                    this->RichTextBox_Loss_Training_PMD->Text = this->_ptr_chart_parameters_loss->struct_training_mean.Get__Past_Mean_Diff().ToString("###0.000000000");
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                    this->RichTextBox_Loss_Validating_Current->Text = this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 1u].ToString("###0.000000000");
                    this->RichTextBox_Loss_Validating_Higher->Text = this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 1u].ToString("###0.000000000");
                    this->RichTextBox_Loss_Validating_Lower->Text = this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 1u].ToString("###0.000000000");
                    this->RichTextBox_Loss_Validating_Mean->Text = this->_ptr_chart_parameters_loss->struct_validating_mean.Get__Current_Mean().ToString("###0.000000000");
                    this->RichTextBox_Loss_Validating_PMD->Text = this->_ptr_chart_parameters_loss->struct_validating_mean.Get__Past_Mean_Diff().ToString("###0.000000000");
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                    this->RichTextBox_Loss_Testing_Current->Text = this->_ptr_chart_parameters_loss->y_current_value[tmp_index_series_shift + 2u].ToString("###0.000000000");
                    this->RichTextBox_Loss_Testing_Higher->Text = this->_ptr_chart_parameters_loss->y_maximum_value[tmp_index_series_shift + 2u].ToString("###0.000000000");
                    this->RichTextBox_Loss_Testing_Lower->Text = this->_ptr_chart_parameters_loss->y_minimum_value[tmp_index_series_shift + 2u].ToString("###0.000000000");
                    this->RichTextBox_Loss_Testing_Mean->Text = this->_ptr_chart_parameters_loss->struct_testing_mean.Get__Current_Mean().ToString("###0.000000000");
                    this->RichTextBox_Loss_Testing_PMD->Text = this->_ptr_chart_parameters_loss->struct_testing_mean.Get__Past_Mean_Diff().ToString("###0.000000000");
                            break;
            }
        }

        System::Void Form_Neural_Network::Output_Accuracy_Information(enum MyEA::Common::ENUM_TYPE_DATASET const type_accuracy_received, unsigned int const index_series_received)
        {
            unsigned int const tmp_index_series_shift(index_series_received * 3u);

            switch(type_accuracy_received)
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                    this->RichTextBox_Accuracy_Training_Current->Text = this->_ptr_chart_parameters_accuracy->y_current_value[tmp_index_series_shift].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Training_Higher->Text = this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Training_Lower->Text = this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Training_Mean->Text = this->_ptr_chart_parameters_accuracy->struct_training_mean.Get__Current_Mean().ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Training_PMD->Text = this->_ptr_chart_parameters_accuracy->struct_training_mean.Get__Past_Mean_Diff().ToString("###0.000000000");
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
                    this->RichTextBox_Accuracy_Validating_Current->Text = this->_ptr_chart_parameters_accuracy->y_current_value[tmp_index_series_shift + 1u].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Validating_Higher->Text = this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 1u].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Validating_Lower->Text = this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 1u].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Validating_Mean->Text = this->_ptr_chart_parameters_accuracy->struct_validating_mean.Get__Current_Mean().ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Validating_PMD->Text = this->_ptr_chart_parameters_accuracy->struct_validating_mean.Get__Past_Mean_Diff().ToString("###0.000000000");
                            break;
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
                    this->RichTextBox_Accuracy_Testing_Current->Text = this->_ptr_chart_parameters_accuracy->y_current_value[tmp_index_series_shift + 2u].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Testing_Higher->Text = this->_ptr_chart_parameters_accuracy->y_maximum_value[tmp_index_series_shift + 2u].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Testing_Lower->Text = this->_ptr_chart_parameters_accuracy->y_minimum_value[tmp_index_series_shift + 2u].ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Testing_Mean->Text = this->_ptr_chart_parameters_accuracy->struct_testing_mean.Get__Current_Mean().ToString("###0.000000000");
                    this->RichTextBox_Accuracy_Testing_PMD->Text = this->_ptr_chart_parameters_accuracy->struct_testing_mean.Get__Past_Mean_Diff().ToString("###0.000000000");
                            break;
            }
        }

        System::Void Form_Neural_Network::Button_Chart_Loss_Reset__Click(System::Object^  sender, System::EventArgs^  e)
        {
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_loss->number_series; ++i)
            {
                this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_" + i.ToString()]->Points->Clear();

                this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_" + i.ToString()]->Points->Clear();

                this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_" + i.ToString()]->Points->Clear();

                this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_TRAINING_" + i.ToString()]->Points->Clear();

                this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_VALIDATING_" + i.ToString()]->Points->Clear();

                this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + i.ToString()]->Points->Clear();
            }
            
            this->Chart_Loss->ChartAreas[0u]->AxisX->ScaleView->ZoomReset();
            
            this->Chart_Loss->ChartAreas[0u]->AxisX->Minimum = 0.0;
            
            this->Chart_Loss->ChartAreas[0u]->AxisX->Maximum = NAN;
            
            this->_ptr_chart_parameters_loss->total_data_point = 0.0;

            this->Chart_Loss->ChartAreas[0u]->RecalculateAxesScale();

            for(unsigned int i(0u); i != this->_ptr_chart_parameters_loss->y_current_value.Count; ++i)
            { this->_ptr_chart_parameters_loss->y_current_value[i] = 0.0; }
            
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_loss->y_minimum_value.Count; ++i)
            { this->_ptr_chart_parameters_loss->y_minimum_value[i] = (std::numeric_limits<double>::max)(); }
            
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_loss->y_maximum_value.Count; ++i)
            { this->_ptr_chart_parameters_loss->y_maximum_value[i] = -(std::numeric_limits<double>::max)(); }
        }
        
        System::Void Form_Neural_Network::Button_Chart_Loss_Difference_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->_use_Datapoint_Difference = !this->_use_Datapoint_Difference;

            this->Button_Chart_Loss_Difference->Text = "Difference (" + (this->_use_Datapoint_Difference ? "true" : "false") + ")";
        }
        
        System::Void Form_Neural_Network::Button_Chart_Loss_Apply_Click(System::Object^  sender, System::EventArgs^  e)
        {
            double const tmp_minimum_value(Convert::ToDouble(this->NumericUpDown_Chart_Loss_Y_Minimum->Value)),
                                tmp_maximum_value(Convert::ToDouble(this->NumericUpDown_Chart_Loss_Y_Maximum->Value));
            
            if(tmp_minimum_value >= tmp_maximum_value
              &&
              tmp_minimum_value != 0.0
              &&
              tmp_maximum_value != 0.0) { return; }

            this->Chart_Loss->ChartAreas[0]->AxisY->Minimum =  tmp_minimum_value == 0.0 ? NAN : tmp_minimum_value;
            
            this->Chart_Loss->ChartAreas[0]->AxisY->Maximum = tmp_maximum_value == 0.0 ? NAN : tmp_maximum_value;
            
            this->Chart_Loss->ChartAreas[0]->AxisY->Interval = (tmp_maximum_value - tmp_minimum_value) / INTERVAL_X;
        }
        
        System::Void Form_Neural_Network::Button_Chart_Loss_Rescale_Click(System::Object^  sender, System::EventArgs^  e)
        {
            double tmp_minimum,
                      tmp_maximum,
                      tmp_minimum_range,
                      tmp_maximum_range,
                      tmp_distance;

            if(this->_ptr_chart_parameters_loss->total_data_point != 0.0)
            {
                tmp_minimum_range = (std::numeric_limits<double>::max)();

                tmp_maximum_range = -(std::numeric_limits<double>::max)();

                for(unsigned int i(0u); i != this->_ptr_chart_parameters_loss->number_series; ++i)
                {
                    // SERIES_LOSS_TRAINING.
                    if(this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                        
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                    }
                    // |END| SERIES_LOSS_TRAINING. |END|
                    
                    // SERIES_LOSS_VALIDATING.
                    if(this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                    }
                    // |END| SERIES_LOSS_VALIDATING. |END|
                    
                    // SERIES_LOSS_TESTING.
                    if(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                    }
                    // |END| SERIES_LOSS_TESTING. |END|

                    // SERIES_LOSS_TESTING_VS_TRAINING.
                    if(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_TRAINING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                    }
                    // |END| SERIES_LOSS_TESTING_VS_TRAINING. |END|
                    
                    // SERIES_LOSS_TESTING_VS_VALIDATING.
                    if(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_VALIDATING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_VALIDATING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_TESTING_VS_VALIDATING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                    }
                    // |END| SERIES_LOSS_TESTING_VS_VALIDATING. |END|
                    
                    // SERIES_LOSS_VALIDATING_VS_TRAINING.
                    if(this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Loss->Series[L"SERIES_LOSS_VALIDATING_VS_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                    }
                    // |END| SERIES_LOSS_VALIDATING_VS_TRAINING. |END|
                }
                
                if(tmp_minimum_range == (std::numeric_limits<double>::max)()) { return; }

                if(tmp_maximum_range == -(std::numeric_limits<double>::max)()) { return; }

                tmp_distance = MyEA::Math::Maximum<double>((tmp_maximum_range - tmp_minimum_range) * 10.0 / 100.0, 0.1);
                
                this->Chart_Loss->ChartAreas[0]->AxisY->Maximum = tmp_maximum_range + tmp_distance;

                this->Chart_Loss->ChartAreas[0]->AxisY->Minimum = tmp_minimum_range - tmp_distance;
            }
        }
        
        System::Void Form_Neural_Network::Button_Chart_Accuracy_Apply_Click(System::Object^  sender, System::EventArgs^  e)
        {
            double const tmp_minimum_value(Convert::ToDouble(this->NumericUpDown_Chart_Accuracy_Y_Minimum->Value)),
                                tmp_maximum_value(Convert::ToDouble(this->NumericUpDown_Chart_Accuracy_Y_Maximum->Value));
            
            if(tmp_minimum_value >= tmp_maximum_value
              &&
              tmp_minimum_value != 0.0
              &&
              tmp_maximum_value != 0.0) { return; }

            this->Chart_Accuracy->ChartAreas[0]->AxisY->Minimum =  tmp_minimum_value == 0.0 ? NAN : tmp_minimum_value;
            
            this->Chart_Accuracy->ChartAreas[0]->AxisY->Maximum = tmp_maximum_value == 0.0 ? NAN : tmp_maximum_value;
            
            this->Chart_Accuracy->ChartAreas[0]->AxisY->Interval = (tmp_maximum_value - tmp_minimum_value) / INTERVAL_X;
        }
        
        System::Void Form_Neural_Network::Button_Chart_Accuracy_Reset__Click(System::Object^  sender, System::EventArgs^  e)
        {
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_accuracy->number_series; ++i)
            {
                this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TRAINING_" + i.ToString()]->Points->Clear();

                this->Chart_Accuracy->Series[L"SERIES_ACCURACY_VALIDATING_" + i.ToString()]->Points->Clear();
                
                this->Chart_Accuracy->Series[L"SERIES_ACCURACY_TESTING_" + i.ToString()]->Points->Clear();
            }
            
            this->Chart_Accuracy->ChartAreas[0u]->AxisX->ScaleView->ZoomReset();
            
            this->Chart_Accuracy->ChartAreas[0u]->AxisX->Minimum = 0.0;
            
            this->Chart_Accuracy->ChartAreas[0u]->AxisX->Maximum = NAN;
            
            this->_ptr_chart_parameters_accuracy->total_data_point = 0.0;

            this->Chart_Accuracy->ChartAreas[0u]->RecalculateAxesScale();

            for(unsigned int i(0u); i != this->_ptr_chart_parameters_accuracy->y_current_value.Count; ++i)
            { this->_ptr_chart_parameters_accuracy->y_current_value[i] = 0.0; }
            
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_accuracy->y_minimum_value.Count; ++i)
            { this->_ptr_chart_parameters_accuracy->y_minimum_value[i] = (std::numeric_limits<double>::max)(); }
            
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_accuracy->y_maximum_value.Count; ++i)
            { this->_ptr_chart_parameters_accuracy->y_maximum_value[i] = -(std::numeric_limits<double>::max)(); }
        }
        
        System::Void Form_Neural_Network::Button_Chart_Output_Apply_Click(System::Object^  sender, System::EventArgs^  e)
        {
            double const tmp_minimum_value(Convert::ToDouble(this->NumericUpDown_Chart_Output_Y_Minimum->Value)),
                                tmp_maximum_value(Convert::ToDouble(this->NumericUpDown_Chart_Output_Y_Maximum->Value));
            
            if(tmp_minimum_value >= tmp_maximum_value
              &&
              tmp_minimum_value != 0.0
              &&
              tmp_maximum_value != 0.0) { return; }

            this->Chart_Output->ChartAreas[0]->AxisY->Minimum =  tmp_minimum_value == 0.0 ? NAN : tmp_minimum_value;
            
            this->Chart_Output->ChartAreas[0]->AxisY->Maximum = tmp_maximum_value == 0.0 ? NAN : tmp_maximum_value;
            
            this->Chart_Output->ChartAreas[0]->AxisY->Interval = (tmp_maximum_value - tmp_minimum_value) / INTERVAL_X;
        }
        
        System::Void Form_Neural_Network::Button_Chart_Output_Reset__Click(System::Object^  sender, System::EventArgs^  e)
        {
            for(unsigned int i(0u); i != this->_ptr_chart_parameters_output->number_series; ++i)
            {
                this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + i.ToString()]->Points->Clear();

                this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + i.ToString()]->Points->Clear();
                
                this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + i.ToString()]->Points->Clear();
            }
            
            this->Chart_Output->ChartAreas[0u]->AxisX->ScaleView->ZoomReset();
            
            this->Chart_Output->ChartAreas[0u]->AxisX->Minimum = 0.0;
            
            this->Chart_Output->ChartAreas[0u]->AxisX->Maximum = NAN;
            
            this->_ptr_chart_parameters_output->y_minimum_value[0] = (std::numeric_limits<double>::max)();
            this->_ptr_chart_parameters_output->y_maximum_value[0] = -(std::numeric_limits<double>::max)();
            this->_ptr_chart_parameters_output->total_data_point = 0.0;

            this->Chart_Output->ChartAreas[0u]->RecalculateAxesScale();
        }

        System::Void Form_Neural_Network::Button_Chart_Grid_Search_Reset__Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points->Clear();

            this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points->Clear();

            this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points->Clear();
            
            this->Chart_Grid_Search->ChartAreas[0u]->AxisX->ScaleView->ZoomReset();
            
            this->Chart_Grid_Search->ChartAreas[0u]->AxisX->Minimum = 0.0;
            
            this->Chart_Grid_Search->ChartAreas[0u]->AxisX->Maximum = NAN;
            
            this->_ptr_chart_parameters_grid_search->total_data_point = 0.0;

            this->Chart_Grid_Search->ChartAreas[0u]->RecalculateAxesScale();

            this->DataGridView_Grid_Search->Rows->Clear();
        }
        
        System::Void Form_Neural_Network::Button_Chart_Grid_Search_Rescale_Click(System::Object^  sender, System::EventArgs^  e)
        {
            double tmp_minimum,
                      tmp_maximum,
                      tmp_minimum_range,
                      tmp_maximum_range,
                      tmp_distance;

            if(this->_ptr_chart_parameters_grid_search->total_data_point != 0.0)
            {
                tmp_minimum_range = (std::numeric_limits<double>::max)();

                tmp_maximum_range = -(std::numeric_limits<double>::max)();

                // SERIES_GRID_SEARCH_TRAINING.
                if(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points->Count != 0)
                {
                    tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points)->YValues[0u];

                    if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                    tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TRAINING"]->Points)->YValues[0u];

                    if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                }
                // |END| SERIES_GRID_SEARCH_TRAINING. |END|
                    
                // SERIES_GRID_SEARCH_VALIDATING.
                if(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points->Count != 0)
                {
                    tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points)->YValues[0u];

                    if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                    tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_VALIDATING"]->Points)->YValues[0u];

                    if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                }
                // |END| SERIES_GRID_SEARCH_VALIDATING. |END|
                    
                // SERIES_GRID_SEARCH_TESTING.
                if(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points->Count != 0)
                {
                    tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points)->YValues[0u];

                    if(tmp_minimum_range > tmp_minimum) { tmp_minimum_range = tmp_minimum; }
                    
                    tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Grid_Search->Series[L"SERIES_GRID_SEARCH_TESTING"]->Points)->YValues[0u];

                    if(tmp_maximum_range < tmp_maximum) { tmp_maximum_range = tmp_maximum; }
                }
                // |END| SERIES_GRID_SEARCH_TESTING. |END|
                
                if(tmp_minimum_range == (std::numeric_limits<double>::max)()) { return; }

                if(tmp_maximum_range == -(std::numeric_limits<double>::max)()) { return; }

                tmp_distance = MyEA::Math::Maximum<double>((tmp_maximum_range - tmp_minimum_range) * 10.0 / 100.0, 0.1);
                
                this->Chart_Grid_Search->ChartAreas[0]->AxisY->Maximum = tmp_maximum_range + tmp_distance;

                this->Chart_Grid_Search->ChartAreas[0]->AxisY->Minimum = tmp_minimum_range - tmp_distance;
            }
        }

        System::Void Form_Neural_Network::Button_Chart_Output_Type_Click(System::Object^  sender, System::EventArgs^  e)
        {
            SeriesChartType const tmp_SeriesChartType(this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_0"]->ChartType == SeriesChartType::FastPoint ? SeriesChartType::FastLine : SeriesChartType::FastPoint);
            
            if(tmp_SeriesChartType == SeriesChartType::FastPoint)
            { this->Button_Chart_Output_Type->Text = "Fast Point"; }
            else
            { this->Button_Chart_Output_Type->Text = "Fast Line"; }

            for(unsigned int i(0u); i != this->_ptr_chart_parameters_output->number_series; ++i)
            {
                this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + i.ToString()]->ChartType = tmp_SeriesChartType;

                this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + i.ToString()]->ChartType = tmp_SeriesChartType;

                this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + i.ToString()]->ChartType = tmp_SeriesChartType;
            }
        }
        
        System::Void Form_Neural_Network::Button_Chart_Output_Rescale_Click(System::Object^  sender, System::EventArgs^  e)
        {
            double tmp_minimum,
                      tmp_maximum,
                      tmp_distance;

            if(this->_ptr_chart_parameters_output->total_data_point != 0.0)
            {
                this->_ptr_chart_parameters_output->y_minimum_value[0] = (std::numeric_limits<double>::max)();

                this->_ptr_chart_parameters_output->y_maximum_value[0] = -(std::numeric_limits<double>::max)();

                for(unsigned int i(0u); i != this->_ptr_chart_parameters_output->number_series; ++i)
                {
                    // SERIES_OUTPUT_TRAINING.
                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(this->_ptr_chart_parameters_output->y_minimum_value[0] > tmp_minimum) { this->_ptr_chart_parameters_output->y_minimum_value[0] = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Output->Series[L"SERIES_OUTPUT_TRAINING_" + i.ToString()]->Points)->YValues[0u];

                        if(this->_ptr_chart_parameters_output->y_maximum_value[0] < tmp_maximum) { this->_ptr_chart_parameters_output->y_maximum_value[0] = tmp_maximum; }
                    }
                    // |END| SERIES_OUTPUT_TRAINING. |END|
                    
                    // SERIES_OUTPUT_VALIDATING.
                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + i.ToString()]->Points)->YValues[0u];

                        if(this->_ptr_chart_parameters_output->y_minimum_value[0] > tmp_minimum) { this->_ptr_chart_parameters_output->y_minimum_value[0] = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Output->Series[L"SERIES_OUTPUT_VALIDATING_" + i.ToString()]->Points)->YValues[0u];

                        if(this->_ptr_chart_parameters_output->y_maximum_value[0] < tmp_maximum) { this->_ptr_chart_parameters_output->y_maximum_value[0] = tmp_maximum; }
                    }
                    // |END| SERIES_OUTPUT_VALIDATING. |END|
                    
                    // SERIES_OUTPUT_TESTING.
                    if(this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + i.ToString()]->Points->Count != 0)
                    {
                        tmp_minimum = this->Chart_Series_Find_Minimum_Y(this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + i.ToString()]->Points)->YValues[0u];

                        if(this->_ptr_chart_parameters_output->y_minimum_value[0] > tmp_minimum) { this->_ptr_chart_parameters_output->y_minimum_value[0] = tmp_minimum; }
                    
                        tmp_maximum = this->Chart_Series_Find_Maximum_Y(this->Chart_Output->Series[L"SERIES_OUTPUT_TESTING_" + i.ToString()]->Points)->YValues[0u];

                        if(this->_ptr_chart_parameters_output->y_maximum_value[0] < tmp_maximum) { this->_ptr_chart_parameters_output->y_maximum_value[0] = tmp_maximum; }
                    }
                    // |END| SERIES_OUTPUT_TESTING. |END|
                }
                
                if(this->_ptr_chart_parameters_output->y_minimum_value[0] == (std::numeric_limits<double>::max)()) { return; }

                if(this->_ptr_chart_parameters_output->y_maximum_value[0] == -(std::numeric_limits<double>::max)()) { return; }

                tmp_distance = MyEA::Math::Maximum<double>((this->_ptr_chart_parameters_output->y_maximum_value[0] - this->_ptr_chart_parameters_output->y_minimum_value[0]) * 10.0 / 100.0, 0.1);
                
                this->Chart_Output->ChartAreas[0]->AxisY->Maximum = this->_ptr_chart_parameters_output->y_maximum_value[0] + tmp_distance;

                this->Chart_Output->ChartAreas[0]->AxisY->Minimum = this->_ptr_chart_parameters_output->y_minimum_value[0] - tmp_distance;
            }
        }

        System::Void Form_Neural_Network::Button_Training_Stop_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Button_Training_Stop->Enabled = false;

            this->_signal_training_stop = true;
        }
        
        System::Void Form_Neural_Network::Button_Training_Menu_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Button_Training_Menu->Enabled = false;

            this->_signal_training_menu = true;
        }

        System::Void Form_Neural_Network::Reset__Signal_Training_Menu(void)
        {
            this->Button_Training_Menu->Enabled = true;

            this->_signal_training_menu = false;
        }

        System::Windows::Forms::DataVisualization::Charting::DataPoint^ Form_Neural_Network::Chart_Series_Find_Maximum_Y(System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received)
        {
            if(series_received->Count == 0) { return(nullptr); }

            System::Windows::Forms::DataVisualization::Charting::DataPoint ^tmp_data_find_maximum_y(series_received[0]);
            
            for(int i(1); i != series_received->Count; ++i)
            {
                if(series_received[i]->YValues[0u] > tmp_data_find_maximum_y->YValues[0u])
                { tmp_data_find_maximum_y = series_received[i]; }
            }

            return(tmp_data_find_maximum_y);
        }
        
        System::Windows::Forms::DataVisualization::Charting::DataPoint^ Form_Neural_Network::Chart_Series_Find_Minimum_Y(System::Windows::Forms::DataVisualization::Charting::DataPointCollection^ series_received)
        {
            if(series_received->Count == 0) { return(nullptr); }

            System::Windows::Forms::DataVisualization::Charting::DataPoint ^tmp_data_find_minimum_y(series_received[0]);

            for(int i(1); i != series_received->Count; ++i)
            {
                if(series_received[i]->YValues[0u] < tmp_data_find_minimum_y->YValues[0u])
                { tmp_data_find_minimum_y = series_received[i]; }
            }

            return(tmp_data_find_minimum_y);
        }
        
        System::Void Form_Neural_Network::Button_Panel_Information_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Panel_Information->Visible = true;
            this->Panel_Chart_Loss->Visible = false;
            this->Panel_Chart_Accuracy->Visible = false;
            this->Panel_Chart_Grid_Search->Visible = false;
            this->Panel_Chart_Output->Visible = false;
        }
        
        System::Void Form_Neural_Network::Button_Panel_Chart_Loss_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Panel_Information->Visible = false;
            this->Panel_Chart_Loss->Visible = true;
            this->Panel_Chart_Accuracy->Visible = false;
            this->Panel_Chart_Grid_Search->Visible = false;
            this->Panel_Chart_Output->Visible = false;
        }
        
        System::Void Form_Neural_Network::Button_Panel_Chart_Accuracy_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Panel_Information->Visible = false;
            this->Panel_Chart_Loss->Visible = false;
            this->Panel_Chart_Accuracy->Visible = true;
            this->Panel_Chart_Grid_Search->Visible = false;
            this->Panel_Chart_Output->Visible = false;
        }
        
        System::Void Form_Neural_Network::Button_Panel_Chart_Output_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Panel_Information->Visible = false;
            this->Panel_Chart_Loss->Visible = false;
            this->Panel_Chart_Accuracy->Visible = false;
            this->Panel_Chart_Grid_Search->Visible = false;
            this->Panel_Chart_Output->Visible = true;
        }
        
        System::Void Form_Neural_Network::Button_Panel_Chart_Grid_Search_Click(System::Object^  sender, System::EventArgs^  e)
        {
            this->Panel_Information->Visible = false;
            this->Panel_Chart_Loss->Visible = false;
            this->Panel_Chart_Accuracy->Visible = false;
            this->Panel_Chart_Grid_Search->Visible = true;
            this->Panel_Chart_Output->Visible = false;
        }
        
        System::Void Form_Neural_Network::Chart_Loss_MouseWheel(System::Object^  sender, MouseEventArgs^  e)
        {
            this->Chart_MouseWheel(this->_ptr_chart_parameters_loss, this->Chart_Loss->ChartAreas[0], e);
        }
        
        System::Void Form_Neural_Network::Chart_Loss_MouseEnter(System::Object^  sender, System::EventArgs^  e) { this->Chart_Loss->Focus(); }

        System::Void Form_Neural_Network::Chart_Loss_MouseLeave(System::Object^  sender, System::EventArgs^  e) { this->Chart_Loss->Parent->Focus(); }
        
        System::Void Form_Neural_Network::Chart_Accuracy_MouseWheel(System::Object^  sender, MouseEventArgs^  e)
        {
            this->Chart_MouseWheel(this->_ptr_chart_parameters_accuracy, this->Chart_Accuracy->ChartAreas[0], e);
        }
        
        System::Void Form_Neural_Network::Chart_Accuracy_MouseEnter(System::Object^  sender, System::EventArgs^  e) { this->Chart_Accuracy->Focus(); }

        System::Void Form_Neural_Network::Chart_Accuracy_MouseLeave(System::Object^  sender, System::EventArgs^  e) { this->Chart_Accuracy->Parent->Focus(); }
        
        System::Void Form_Neural_Network::Chart_Output_MouseWheel(System::Object^  sender, MouseEventArgs^  e)
        {
            this->Chart_MouseWheel(this->_ptr_chart_parameters_output, this->Chart_Output->ChartAreas[0], e);
        }
        
        System::Void Form_Neural_Network::Chart_Output_MouseEnter(System::Object^  sender, System::EventArgs^  e) { this->Chart_Output->Focus(); }

        System::Void Form_Neural_Network::Chart_Output_MouseLeave(System::Object^  sender, System::EventArgs^  e) { this->Chart_Output->Parent->Focus(); }
        
        System::Void Form_Neural_Network::Chart_Grid_Search_MouseWheel(System::Object^  sender, MouseEventArgs^  e)
        {
            this->Chart_MouseWheel(this->_ptr_chart_parameters_grid_search, this->Chart_Grid_Search->ChartAreas[0], e);
        }
        
        System::Void Form_Neural_Network::Chart_Grid_Search_MouseEnter(System::Object^  sender, System::EventArgs^  e) { this->Chart_Grid_Search->Focus(); }

        System::Void Form_Neural_Network::Chart_Grid_Search_MouseLeave(System::Object^  sender, System::EventArgs^  e) { this->Chart_Grid_Search->Parent->Focus(); }
        
        System::Void Form_Neural_Network::Chart_MouseWheel(ref struct struct_Chart_Parameters ^ptr_chart_parameters_received, System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received, MouseEventArgs^  e)
        {
            double const tmp_x_minimum = ptr_chart_areas_received->AxisX->Minimum,
                                tmp_x_maximum = ptr_chart_areas_received->AxisX->Maximum,
                                tmp_x_view_minimum = ptr_chart_areas_received->AxisX->ScaleView->ViewMinimum,
                                tmp_x_view_maximum = ptr_chart_areas_received->AxisX->ScaleView->ViewMaximum,
                                tmp_x_location = System::Math::Round(ptr_chart_areas_received->AxisX->PixelPositionToValue(e->Location.X));
            double tmp_x_start,
                      tmp_x_end;

            if(e->Delta < 0)
            {
                if(ptr_chart_areas_received->AxisX->ScaleView->ViewMinimum == tmp_x_minimum
                   &&
                   ptr_chart_areas_received->AxisX->ScaleView->ViewMaximum == tmp_x_maximum)
                { return; }
                
                tmp_x_start = tmp_x_view_minimum - System::Math::Round((tmp_x_location - tmp_x_view_minimum) / 4);
                tmp_x_end = tmp_x_view_maximum + System::Math::Round((tmp_x_view_maximum - tmp_x_location) / 4);

                tmp_x_start = MyEA::Math::Clip<double>(tmp_x_start, tmp_x_minimum, tmp_x_maximum);
                tmp_x_end = MyEA::Math::Clip<double>(tmp_x_end, tmp_x_minimum, tmp_x_maximum);
                
                ptr_chart_areas_received->AxisX->ScaleView->Zoom(tmp_x_start, tmp_x_end);
                
                if(ptr_chart_areas_received->AxisX->ScaleView->ViewMinimum <= tmp_x_minimum
                   &&
                   ptr_chart_areas_received->AxisX->ScaleView->ViewMaximum >= tmp_x_maximum)
                { ptr_chart_areas_received->AxisX->ScaleView->ZoomReset(); }
            }
            else if(e->Delta > 0)
            {
                tmp_x_start = tmp_x_view_minimum + System::Math::Floor((tmp_x_location - tmp_x_view_minimum) / 5);
                tmp_x_end = tmp_x_view_maximum - System::Math::Floor((tmp_x_view_maximum - tmp_x_location) / 5);
                
                if(tmp_x_start == tmp_x_minimum
                   &&
                   tmp_x_end == tmp_x_maximum)
                { return; }

                tmp_x_start = MyEA::Math::Clip<double>(tmp_x_start, tmp_x_minimum, tmp_x_maximum);
                tmp_x_end = MyEA::Math::Clip<double>(tmp_x_end, tmp_x_minimum, tmp_x_maximum);
                
                ptr_chart_areas_received->AxisX->ScaleView->Zoom(tmp_x_start, tmp_x_end);
            }
            
            this->ScaleView_Interval(ptr_chart_areas_received);

            this->Compute_Precision(ptr_chart_parameters_received, ptr_chart_areas_received);

            ptr_chart_parameters_received->block_view_X = ptr_chart_areas_received->AxisX->ScaleView->ViewMaximum - ptr_chart_areas_received->AxisX->ScaleView->ViewMinimum;
        }
        
        System::Void Form_Neural_Network::ScaleView_Interval(System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received)
        {
            double const tmp_minimum_value(ptr_chart_areas_received->AxisX->ScaleView->ViewMinimum),
                                tmp_maximum_value(ptr_chart_areas_received->AxisX->ScaleView->ViewMaximum);

            ptr_chart_areas_received->AxisX->Interval = ceil((tmp_maximum_value - tmp_minimum_value) / INTERVAL_X);
        }
        
        System::Void Form_Neural_Network::ScaleView_Zoom(double const index_end_received,
                                                                                         ref struct struct_Chart_Parameters ^ptr_chart_parameters_received,
                                                                                         System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received)
        {
            double const tmp_x_minimum = ptr_chart_areas_received->AxisX->Minimum,
                                tmp_x_maximum = ptr_chart_areas_received->AxisX->Maximum,
                                tmp_x_view_minimum = ptr_chart_areas_received->AxisX->ScaleView->ViewMinimum,
                                tmp_x_view_maximum = ptr_chart_areas_received->AxisX->ScaleView->ViewMaximum;
            double tmp_x_start,
                      tmp_x_end;
            
            if(tmp_x_maximum == tmp_x_view_maximum && tmp_x_minimum == tmp_x_view_minimum) { return; }

            tmp_x_end = MyEA::Math::Minimum<double>(tmp_x_maximum, index_end_received + 1.0);
            tmp_x_start = tmp_x_end - ptr_chart_parameters_received->block_view_X;

            ptr_chart_areas_received->AxisX->ScaleView->Zoom(tmp_x_start, tmp_x_end);
        }
        
        System::Void Form_Neural_Network::Compute_Precision(ref struct struct_Chart_Parameters ^ptr_chart_parameters_received, System::Windows::Forms::DataVisualization::Charting::ChartArea^  ptr_chart_areas_received)
        {
            double const tmp_minimum_value(ptr_chart_areas_received->AxisY->ScaleView->ViewMinimum),
                                tmp_maximum_value(ptr_chart_areas_received->AxisY->ScaleView->ViewMaximum);

            ptr_chart_parameters_received->precision = (tmp_maximum_value - tmp_minimum_value) / 100.0;
        }
        
        Form_Neural_Network::~Form_Neural_Network(void)
        {
            PRINT_FORMAT("%s" NEW_LINE, __FUNCTION__);
            SAFE_DELETE(this->components);

            // Main.
            SAFE_DELETE(this->Panel_Button);
            SAFE_DELETE(this->Button_Panel_Chart_Loss);
            SAFE_DELETE(this->Button_Panel_Chart_Accuracy);
            // |END| Main. |END|

            // Panel loss.
            SAFE_DELETE(this->_ptr_chart_parameters_loss);
            SAFE_DELETE(this->Panel_Chart_Loss);
            SAFE_DELETE(this->Chart_Loss);
            SAFE_DELETE(this->Button_Training_Stop);
            SAFE_DELETE(this->Button_Chart_Loss_Reset);
            SAFE_DELETE(this->Label_Loss_Mean);
            // |END| Panel loss. |END|

            // Panel accuracy.
            SAFE_DELETE(this->_ptr_chart_parameters_accuracy);
            SAFE_DELETE(this->Panel_Chart_Accuracy);
            SAFE_DELETE(this->Chart_Accuracy);
            SAFE_DELETE(this->Button_Chart_Accuracy_Reset);
            SAFE_DELETE(this->Label_Accuracy_Mean);
            // |END| Panel accuracy. |END|

            // Panel output.
            SAFE_DELETE(this->_ptr_chart_parameters_output);
            SAFE_DELETE(this->Panel_Chart_Output);
            SAFE_DELETE(this->Chart_Output);
            SAFE_DELETE(this->Button_Chart_Output_Reset);
            // |END| Panel output. |END|

            // Panel grid search.
            SAFE_DELETE(this->_ptr_chart_parameters_grid_search);
            SAFE_DELETE(this->Panel_Chart_Grid_Search);
            SAFE_DELETE(this->Chart_Grid_Search);
            SAFE_DELETE(this->Button_Chart_Grid_Search_Reset);
            // |END| Panel grid search. |END|
        }
    }
}