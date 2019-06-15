#include "stdafx.hpp"

#include <Expert/CBase.hpp>

namespace MyEA
{
    namespace Common
    {
        CBase::CBase(void) : p_trailing_mode(-1),
                                        p_instanTdeal(-1),
                                        p_debug_box(true),
                                        p_debug_trade(false),
                                        p_every_tick(false),
                                        p_expiration_close_position(false),
                                        p_async_mode(true),
                                        p_expiration_bars_pending(2),
                                        p_expiration_bars_position(2),
                                        p_threshold_close(75),
                                        p_threshold_open(100),
                                        p_Margin(0.0),
                                        p_Risk(0.0),
                                        p_Lots(0.1),
                                        p_Stop_Level(20.0),
                                        p_Take_Level(20.0),
                                        p_Price_Level(20.0),
                                        p_deviation_point(3),
                                        p_magic(438194249143680291),
                                        p_title("MyEA"),
                                        p_accounTmode_trade(ACCOUNT_MODE_TRADE_NONE),
                                        p_iniTphase(INITPHASE_AWAKE),
                                        p_type_order_filling(ORDER_FILLING_FOK),
                                        p_time_frames(TIME_FRAMES_PERIOD_M5),
                                        p_strcut_time_frames()
        {
        }

        CBase::~CBase(void)
        {
        }

        const bool CBase::Initialize(const unsigned char type_order_filling_received,
                                                const signed char trailing_mode_received,
                                                const signed char instanTdeal_received,
                                                const bool debug_box_received,
                                                const bool debug_trade_received,
                                                const bool every_tick_received,
                                                const bool expiration_close_position_received,
                                                const bool async_mode_received,
                                                const unsigned short expiration_bars_pending_received,
                                                const unsigned short expiration_bars_position_received,
                                                const unsigned int threshold_close_received,
                                                const unsigned int threshold_open_received,
                                                const unsigned int time_frames_received,
                                                const double margin_received,
                                                const double risk_received,
                                                const double lots_received,
                                                const double stop_level_received,
                                                const double take_level_received,
                                                const double price_level_received,
                                                const double trailing_stop_level_received,
                                                const double trailing_take_level_received,
                                                const unsigned long long deviation_poinTreceived,
                                                const unsigned long long magic_received,
                                                const std::string title_received,
                                                const std::string symbol_name_received)
        {
            if(this->Get__Init_Phase() != INITPHASE_AWAKE) { return(false); }
            else
            {
                if(!this->Set__Type_Order_Filling(type_order_filling_received)) { MyEA::Common::Message_Box__OK("!this->Set__Type_Order_Filling", "ERROR"); return(false); }
                else if(!this->Set__Trailing_Mode(trailing_mode_received)) { MyEA::Common::Message_Box__OK("!this->Set__Trailing_Mode", "ERROR"); return(false); }
                else if(!this->Set__InstanTDeal(instanTdeal_received)) { MyEA::Common::Message_Box__OK("!this->Set__InstanTDeal", "ERROR"); return(false); }
                else if(!this->Set__Debug_Box(debug_box_received)) { MyEA::Common::Message_Box__OK("!this->Set__Debug_Box", "ERROR"); return(false); }
                else if(!this->Set__Debug_Description(debug_trade_received)) { MyEA::Common::Message_Box__OK("!this->Set__Debug_Description", "ERROR"); return(false); }
                else if(!this->Set__Every_Tick(every_tick_received)) { MyEA::Common::Message_Box__OK("!this->Set__Every_Tick", "ERROR"); return(false); }
                else if(!this->Set__Expiration_Close_Position(expiration_close_position_received)) { MyEA::Common::Message_Box__OK("!this->Set__Expiration_Close_Position", "ERROR"); return(false); }
                else if(!this->Set__Async_Mode(async_mode_received)) { MyEA::Common::Message_Box__OK("!this->Set__Async_Mode", "ERROR"); return(false); }
                else if(!this->Set__Expiration_Bars_Pending(expiration_bars_pending_received)) { MyEA::Common::Message_Box__OK("!this->Set__Expiration_Bars_Pending", "ERROR"); return(false); }
                else if(!this->Set__Expiration_Bars_Position(expiration_bars_position_received)) { MyEA::Common::Message_Box__OK("!this->Set__Expiration_Bars_Position", "ERROR"); return(false); }
                else if(!this->Set__Threshold_Close(threshold_close_received)) { MyEA::Common::Message_Box__OK("!this->Set__Threshold_Close", "ERROR"); return(false); }
                else if(!this->Set__Threshold_Open(threshold_open_received)) { MyEA::Common::Message_Box__OK("!this->Set__Threshold_Open", "ERROR"); return(false); }
                else if(!this->Set__Margin(margin_received)) { MyEA::Common::Message_Box__OK("!this->Set__Margin", "ERROR"); return(false); }
                else if(!this->Set__Risk(risk_received)) { MyEA::Common::Message_Box__OK("!this->Set__Risk", "ERROR"); return(false); }
                else if(!this->Set__Lots(lots_received)) { MyEA::Common::Message_Box__OK("!this->Set__Lots", "ERROR"); return(false); }
                else if(!this->Set__Stop_Level(stop_level_received)) { MyEA::Common::Message_Box__OK("!this->Set__Stop_Level", "ERROR"); return(false); }
                else if(!this->Set__Take_Level(take_level_received)) { MyEA::Common::Message_Box__OK("!this->Set__Take_Level", "ERROR"); return(false); }
                else if(!this->Set__Price_Level(price_level_received)) { MyEA::Common::Message_Box__OK("!this->Set__Price_Level", "ERROR"); return(false); }
                else if(!this->Set__Trailing_Stop_Level(trailing_stop_level_received)) { MyEA::Common::Message_Box__OK("!this->Set__Trailing_Stop_Level", "ERROR"); return(false); }
                else if(!this->Set__Trailing_Take_Level(trailing_take_level_received)) { MyEA::Common::Message_Box__OK("!this->Set__Trailing_Take_Level", "ERROR"); return(false); }
                else if(!this->Set__Deviation_Point(deviation_poinTreceived)) { MyEA::Common::Message_Box__OK("!this->Set__Deviation_Point", "ERROR"); return(false); }
                else if(!this->Set__Magic(magic_received)) { MyEA::Common::Message_Box__OK("!this->Set__Magic", "ERROR"); return(false); }
                else if(!this->Set__Title(title_received)) { MyEA::Common::Message_Box__OK("!this->Set__Title", "ERROR"); return(false); }
                else if(!this->p_symbol_info.Set__Name(symbol_name_received)) { MyEA::Common::Message_Box__OK("!this->Set__Symbol_Name", "ERROR"); return(false); }
                else if(!this->Init_Time_Frames(time_frames_received)) { MyEA::Common::Message_Box__OK("!this->Init_Time_Frames", "ERROR"); return(false); }
                this->Set__Init_Phase(INITPHASE_TUNING);
                return(true);
            }
        }

        const bool CBase::ValidationSettings(void)
        {
            if(this->Get__Init_Phase() != INITPHASE_TUNING)
            {
                MyEA::Common::Message_Box__OK("{ this->Get__Init_Phase() != INITPHASE_TUNING } .", "CBase::ValidationSettings");
                return(false);
            }
            else
            {
                this->Set__Init_Phase(INITPHASE_VALIDATION);
                this->Set__Init_Phase(INITPHASE_COMPLETE);
                return(true);
            }
        }

        const bool CBase::Init_Time_Frames(const unsigned int time_frames_received)
        {
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M1 = 60;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M2 = 120;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M3 = 180;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M4 = 240;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M5 = 300;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M6 = 360;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M10 = 600;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M12 = 720;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M15 = 900;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M20 = 1200;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_M30 = 1800;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H1 = 3600;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H2 = 7200;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H3 = 10800;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H4 = 14400;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H6 = 21600;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H8 = 28800;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_H12 = 43200;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_D1 = 86400;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_W1 = 604800;
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_MN1 = 2627942; // ~
            return(this->Set__struct_Time_Frames(time_frames_received));
        }

        const bool CBase::Set__Account_Mode_Trade(const unsigned char accounTmode_trade_received)
        {
            switch(accounTmode_trade_received)
            {
                case 0: this->p_accounTmode_trade = ACCOUNT_MODE_TRADE_NONE; return(true);
                case 1: this->p_accounTmode_trade = ACCOUNT_MODE_TRADE_CONTEST; return(true);
                case 2: this->p_accounTmode_trade = ACCOUNT_MODE_TRADE_REAL; return(true);
                case 3: this->p_accounTmode_trade = ACCOUNT_MODE_TRADE_DEMO; return(true);
                default: return(false);
            }
        }

        const bool CBase::Set__struct_Time_Frames(const unsigned int time_frames_received)
        {
            switch(time_frames_received)
            {
                case 60: p_time_frames =  TIME_FRAMES_PERIOD_M1; break;
                case 120: p_time_frames =  TIME_FRAMES_PERIOD_M2; break;
                case 180: p_time_frames =  TIME_FRAMES_PERIOD_M3; break;
                case 240: p_time_frames =  TIME_FRAMES_PERIOD_M4; break;
                case 300: p_time_frames =  TIME_FRAMES_PERIOD_M5; break;
                case 360: p_time_frames =  TIME_FRAMES_PERIOD_M6; break;
                case 600: p_time_frames =  TIME_FRAMES_PERIOD_M10; break;
                case 720: p_time_frames =  TIME_FRAMES_PERIOD_M12; break;
                case 900: p_time_frames =  TIME_FRAMES_PERIOD_M15; break;
                case 1200: p_time_frames =  TIME_FRAMES_PERIOD_M20; break;
                case 1800: p_time_frames =  TIME_FRAMES_PERIOD_M30; break;
                case 3600: p_time_frames =  TIME_FRAMES_PERIOD_H1; break;
                case 7200: p_time_frames =  TIME_FRAMES_PERIOD_H2; break;
                case 10800: p_time_frames =  TIME_FRAMES_PERIOD_H3; break;
                case 14400: p_time_frames =  TIME_FRAMES_PERIOD_H4; break;
                case 21600: p_time_frames =  TIME_FRAMES_PERIOD_H6; break;
                case 28800: p_time_frames =  TIME_FRAMES_PERIOD_H8; break;
                case 43200: p_time_frames =  TIME_FRAMES_PERIOD_H12; break;
                case 86400: p_time_frames =  TIME_FRAMES_PERIOD_D1; break;
                case 604800: p_time_frames =  TIME_FRAMES_PERIOD_W1; break;
                case 2627942: p_time_frames = TIME_FRAMES_PERIOD_MN1; break;
                default: return(false);
            }
            
            this->p_strcut_time_frames.TIME_FRAMES_PERIOD_CURRENT = time_frames_received;

            return(true);
        }

        const std::string CBase::Get__Info(void)
        {
            std::string tmp_info = "[CBase::Info]";
            tmp_info += "\r\n\tTrailing Mode : "+std::to_string(this->Get__Trailing_Mode());
            tmp_info += "\r\n\tInstant Deal : "+std::to_string(this->Get__InstanTDeal());
            tmp_info += "\r\n\tDebug Box : "; tmp_info +=     this->Get__Debug_Box() ? "TRUE" : "FALSE";
            tmp_info += "\r\n\tDebug Description : "; tmp_info +=     this->Get__Debug_Description() ? "TRUE" : "FALSE";
            tmp_info += "\r\n\tEvery Tick : "; tmp_info +=     this->Get__Every_Tick() ? "TRUE" : "FALSE";
            tmp_info += "\r\n\tExpiration Close Position : "; tmp_info += this->Get__Expiration_Close_Position() ? "TRUE" : "FALSE";
            tmp_info += "\r\n\tAsync Mode : "; tmp_info += this->Get__Async_Mode() ? "TRUE" : "FALSE";
            tmp_info += "\r\n\tExpiration Bars Pending : "+std::to_string(this->Get__Expiration_Bars_Pending());
            tmp_info += "\r\n\tExpiration Bars Position : "+std::to_string(this->Get__Expiration_Bars_Position());
            tmp_info += "\r\n\tThreshold Close : "+std::to_string(this->Get__Threshold_Close());
            tmp_info += "\r\n\tThreshold Open : "+std::to_string(this->Get__Threshold_Open());
            tmp_info += "\r\n\tSeconds Time Frames : "+std::to_string(this->Get__struct_Time_Frames());
            tmp_info += "\r\n\tMargin : "+std::to_string(this->Get__Margin());
            tmp_info += "\r\n\tRisk : "+std::to_string(this->Get__Risk());
            tmp_info += "\r\n\tLots : "+std::to_string(this->Get__Lots());
            tmp_info += "\r\n\tStop Level : "+std::to_string(this->Get__Stop_Level());
            tmp_info += "\r\n\tTake Level : "+std::to_string(this->Get__Take_Level());
            tmp_info += "\r\n\tPrice Level : "+std::to_string(this->Get__Price_Level());
            tmp_info += "\r\n\tTrailing Stop Level : "+std::to_string(this->Get__Trailing_Stop_Level());
            tmp_info += "\r\n\tTrailing Take Level : "+std::to_string(this->Get__Trailing_Take_Level());
            tmp_info += "\r\n\tDeviation Point : "+std::to_string(this->Get__Deviation_Point());
            tmp_info += "\r\n\tMagic : "+std::to_string(this->Get__Magic());
            tmp_info += "\r\n\tTitle : "+this->Get__Title();
            tmp_info += "\r\n\tSymbol Name : "+this->Get__Ptr_CSymbolInfo()->Get__Name();
            tmp_info += "\r\n\tType Order Filling : "+ENUM_TYPE_ORDER_FILLING_NAMES[this->Get__Type_Order_Filling()];
            tmp_info += "\r\n\tAccount Trade Mode : "+ENUM_ACCOUNT_MODE_TRADE_NAMES[this->Get__Account_Mode_Trade()];
            tmp_info += "\r\n\tInit_ Phase : "+ENUM_INITPHASE_NAMES[this->Get__Init_Phase()];
            tmp_info += "\r\n\tTime Frames : "+ENUM_TIME_FRAMES_NAMES[this->Get__Time_Frames()];
            return(tmp_info);
        }
    }
}
