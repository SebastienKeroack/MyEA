#pragma once

#include <Expert/CBase.hpp>

namespace MyEA
{
    namespace Common
    {
        class CExpert : public CBase
        {
            protected:
                unsigned short p_Simultaneously_Position;

            public:
                CExpert(void);
                ~CExpert(void);

                // [     SET      ]
                const bool Set__Simultaneously_Position(const unsigned short simultaneously_position_received) { p_Simultaneously_Position = simultaneously_position_received; return(true); }
                // ----- SET -----

                // [     GET      ]
                const unsigned short Get__Simultaneously_Position(void) { return(p_Simultaneously_Position); }

                const std::string Get__Info(void);
                // ----- GET -----

                const bool Initialize(const unsigned char type_order_filling_received,
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
                                        const std::string symbol_name_received);
                const bool ValidationSettings(void);
        };
    }
}