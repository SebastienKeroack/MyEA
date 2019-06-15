#pragma once

#include <string>

#include <Enums/Enum_Event_Signal_Wait.hpp>
#include <Enums/Enum_Account_Mode_Trade.hpp>
#include <Enums/Enum_Init_Phase.hpp>
#include <Enums/Enum_Type_Order_Filling.hpp>
#include <Enums/Enum_Time_Frames.hpp>

#include <Tools/Message_Box.hpp>
#include <CObject.hpp>
#include <Trade/CAccountInfo.hpp>
#include <Trade/CSymbolInfo.hpp>

namespace MyEA
{
    namespace Common
    {
        class CBase : public CObject
        {
            protected:
                signed char p_trailing_mode = 0;
                signed char p_instanTdeal = 0;

                const bool Init_Time_Frames(const unsigned int time_frames_received);
                bool p_debug_box = false;
                bool p_debug_trade = false;
                bool p_every_tick = false;
                bool p_expiration_close_position = false;
                bool p_async_mode = false;

                unsigned short p_expiration_bars_pending = 0u;
                unsigned short p_expiration_bars_position = 0u;

                unsigned int p_threshold_close = 0u;
                unsigned int p_threshold_open = 0u;

                double p_Margin = 0.0;
                double p_Risk = 0.0;
                double p_Lots = 0.0;
                double p_Stop_Level = 0.0;
                double p_Take_Level = 0.0;
                double p_Price_Level = 0.0;
                double p_Trailing_Stop_Level = 0.0;
                double p_Trailing_Take_Level = 0.0;

                unsigned long long p_deviation_point = 0ull;
                unsigned long long p_magic = 0ull;

                std::string p_title;

                enum ENUM_ACCOUNT_MODE_TRADE p_accounTmode_trade;
                enum ENUM_INITPHASE p_iniTphase;
                enum ENUM_TYPE_ORDER_FILLING p_type_order_filling;
                enum ENUM_TIME_FRAMES p_time_frames;

                TIME_FRAMES p_strcut_time_frames;

                CAccountInfo p_accounTinfo;
                CSymbolInfo p_symbol_info;

            public:
                CBase(void);
                ~CBase(void);

                // [     SET      ]
                void Set__Account_Mode_Trade(const ENUM_ACCOUNT_MODE_TRADE accounTmode_trade_received) { p_accounTmode_trade = accounTmode_trade_received; }
                void Set__Init_Phase(const ENUM_INITPHASE iniTphase_received) { p_iniTphase = iniTphase_received; }
                void Set__Time_Frames(const ENUM_TIME_FRAMES time_frames_received) { p_time_frames = time_frames_received; }

                const bool Set__Type_Order_Filling(const unsigned char type_order_filling_received) { p_type_order_filling = static_cast<ENUM_TYPE_ORDER_FILLING>(type_order_filling_received+1); return(true); }
                const bool Set__Account_Mode_Trade(const unsigned char accounTmode_trade_received);
                const bool Set__Trailing_Mode(const signed char trailing_Mode_received) { p_trailing_mode = trailing_Mode_received; return(true); }
                const bool Set__InstanTDeal(const signed char instanTDeal_received) { p_instanTdeal = instanTDeal_received; return(true); }
                const bool Set__Debug_Box(const bool debug_box_received) { p_debug_box = debug_box_received; return(true); }
                const bool Set__Debug_Description(const bool debug_Information_received) { p_debug_trade = debug_Information_received; return(true); }
                const bool Set__Every_Tick(const bool every_tick_received) { p_every_tick = every_tick_received; return(true); }
                const bool Set__Expiration_Close_Position(const bool expiration_close_position_received) { p_expiration_close_position = expiration_close_position_received; return(true); }
                const bool Set__Async_Mode(const bool async_Mode_received) { p_async_mode = async_Mode_received; return(true); }
                const bool Set__Expiration_Bars_Pending(const unsigned short expiration_bars_pending_received) { p_expiration_bars_pending = expiration_bars_pending_received; return(true); }
                const bool Set__Expiration_Bars_Position(const unsigned short expiration_bars_position_received) { p_expiration_bars_position = expiration_bars_position_received; return(true); }
                const bool Set__Threshold_Close(const unsigned int threshold_close_received) { p_threshold_close = threshold_close_received; return(true); }
                const bool Set__Threshold_Open(const unsigned int threshold_open_received) { p_threshold_open = threshold_open_received; return(true); }
                const bool Set__struct_Time_Frames(const unsigned int time_frames_received);
                const bool Set__Margin(const double margin_Received) { p_Margin = margin_Received; return(true); }
                const bool Set__Risk(const double risk_received) { p_Risk = risk_received; return(true); }
                const bool Set__Lots(const double lots_received) { p_Lots = lots_received; return(true); }
                const bool Set__Stop_Level(const double stop_level_received) { p_Stop_Level = stop_level_received; return(true); }
                const bool Set__Take_Level(const double take_level_received) { p_Take_Level = take_level_received; return(true); }
                const bool Set__Price_Level(const double price_level_received) { p_Price_Level = price_level_received; return(true); }
                const bool Set__Trailing_Stop_Level(const double trailing_stop_level_received) { p_Trailing_Stop_Level = trailing_stop_level_received; return(true); }
                const bool Set__Trailing_Take_Level(const double trailing_take_level_received) { p_Trailing_Take_Level = trailing_take_level_received; return(true); }
                const bool Set__Deviation_Point(const unsigned long long deviation_poinTreceived) { p_deviation_point = deviation_poinTreceived; return(true); }
                const bool Set__Magic(const unsigned long long magic_received) { p_magic = magic_received; return(true); }
                const bool Set__Title(const std::string title_received) { p_title = title_received; return(true); }
                // ----- SET -----

                // [     GET      ]
                const signed char Get__Trailing_Mode(void) const { return(p_trailing_mode); }
                const signed char Get__InstanTDeal(void) const { return(p_instanTdeal); }

                const bool Get__Debug_Box(void) const { return(p_debug_box); }
                const bool Get__Debug_Description(void) const { return(p_debug_trade); }
                const bool Get__Every_Tick(void) const { return(p_every_tick); }
                const bool Get__Expiration_Close_Position(void) const { return(p_expiration_close_position); }
                const bool Get__Async_Mode(void) const { return(p_async_mode); }

                const unsigned short Get__Expiration_Bars_Pending(void) const { return(p_expiration_bars_pending); }
                const unsigned short Get__Expiration_Bars_Position(void) const { return(p_expiration_bars_position); }

                const unsigned int Get__Threshold_Close(void) const { return(p_threshold_close); }
                const unsigned int Get__Threshold_Open(void) const { return(p_threshold_open); }
                const unsigned int Get__struct_Time_Frames(void) const { return(p_strcut_time_frames.TIME_FRAMES_PERIOD_CURRENT); }

                const double Get__Margin(void) const { return(p_Margin); }
                const double Get__Risk(void) const { return(p_Risk); }
                const double Get__Lots(void) const { return(p_Lots); }
                const double Get__Stop_Level(void) const { return(p_Stop_Level); }
                const double Get__Take_Level(void) const { return(p_Take_Level); }
                const double Get__Price_Level(void) const { return(p_Price_Level); }
                const double Get__Trailing_Stop_Level(void) const { return(p_Trailing_Stop_Level); }
                const double Get__Trailing_Take_Level(void) const { return(p_Trailing_Take_Level); }

                const unsigned long long Get__Deviation_Point(void) const { return(p_deviation_point); }
                const unsigned long long Get__Magic(void) const { return(p_magic); }

                const std::string Get__Title(void) const { return(p_title); }

                const ENUM_ACCOUNT_MODE_TRADE Get__Account_Mode_Trade(void) const { return(p_accounTmode_trade); }
                const ENUM_INITPHASE Get__Init_Phase(void) const { return(p_iniTphase); }
                const ENUM_TYPE_ORDER_FILLING Get__Type_Order_Filling(void) const { return(p_type_order_filling); }
                const ENUM_TIME_FRAMES Get__Time_Frames(void) const { return(p_time_frames); }

                const std::string Get__Info(void);

                CAccountInfo* Get__Ptr_CAccountInfo(void) { return(&p_accounTinfo); }
                CSymbolInfo* Get__Ptr_CSymbolInfo(void) { return(&p_symbol_info); }
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
