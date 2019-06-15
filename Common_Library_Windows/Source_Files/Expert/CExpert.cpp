#include "stdafx.hpp"

#include <Expert/CExpert.hpp>

namespace MyEA
{
    namespace Common
    {
        CExpert::CExpert(void) : p_Simultaneously_Position(1)
        {
        }

        const bool CExpert::Initialize(const unsigned char type_order_filling_received,
                                            const signed char trailing_mode_received,
                                            const signed char instanTdeal_received,
                                            const bool debug_box_received,
                                            const bool debug_trade_received,
                                            const bool every_tick_received,
                                            const bool expiration_close_position_received,
                                            const bool async_mode_received,
                                            const unsigned short expiration_bars_pending_received,
                                            const unsigned short expiration_bars_position_received,
                                            const unsigned short simultaneously_position_received,
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
            if(!this->CBase::Initialize(type_order_filling_received,
                                                    trailing_mode_received,
                                                    instanTdeal_received,
                                                    debug_box_received,
                                                    debug_trade_received,
                                                    every_tick_received,
                                                    expiration_close_position_received,
                                                    async_mode_received,
                                                    expiration_bars_pending_received,
                                                    expiration_bars_position_received,
                                                    threshold_close_received,
                                                    threshold_open_received,
                                                    time_frames_received,
                                                    margin_received,
                                                    risk_received,
                                                    lots_received,
                                                    stop_level_received,
                                                    take_level_received,
                                                    price_level_received,
                                                    trailing_stop_level_received,
                                                    trailing_take_level_received,
                                                    deviation_poinTreceived,
                                                    magic_received,
                                                    title_received,
                                                    symbol_name_received))
            { MyEA::Common::Message_Box__OK("!this->CExpert::Initialize", "ERROR"); return(false); }
            else
            {
                if(!this->Set__Simultaneously_Position(simultaneously_position_received)) { MyEA::Common::Message_Box__OK("!this->Set__Simultaneously_Position", "ERROR"); return(false); }
                else { return(true); }
            }
        }

        const bool CExpert::ValidationSettings(void)
        {
            if(!this->CBase::ValidationSettings()) { MyEA::Common::Message_Box__OK("{ !this->CBase::ValidationSettings() } .", "CExpert::ValidationSettings"); return(false); }
            else { return(true); }
        }

        const std::string CExpert::Get__Info(void)
        {
            std::string tmp_info = "[CExpert::Info]";
            tmp_info += "\r\n\tSimultaneously Position : "+std::to_string(this->Get__Simultaneously_Position());
            tmp_info += "\r\n\t"+this->CBase::Get__Info();
            return(tmp_info);
        }

        CExpert::~CExpert(void)
        {
        }
    }
}
