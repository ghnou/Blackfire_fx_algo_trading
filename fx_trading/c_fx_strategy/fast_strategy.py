from __future__ import (absolute_import, division, print_function, absolute_import, unicode_literals)
import argparse
import fx_trading.utils.constants as cst
import pandas as pd
import numpy as np
from numba import njit, prange
import talib
from pathlib import Path
from fx_trading.utils.data_interact import read_data
from time import time

from fx_trading.utils.parallel_tasks import AsyncMP

RAW_DATA_PATH = str(Path(__file__).parent)
from ta import add_momentum_ta, add_trend_ta, add_volatility_ta, add_others_ta, add_all_ta_features

RESULT_COLS = {cst.T_TIMESTAMP: 'TIMESTAMP', cst.T_ORDER_TYPE: 'ORDER_TYPE', cst.T_SPREAD: 'SPREAD',
               cst.T_PRICE: 'PRICE', cst.T_ENTRY: 'ENTRY', cst.T_SL: 'SL', cst.T_TP: 'TP',
               cst.T_TRADE_STATUS: 'TRADE_STATUS', cst.T_DT_OPEN: 'DT_OPEN', cst.T_DT_CLOSE: 'DT_CLOSE',
               cst.T_EXEC_ENTRY: 'EXEC_ENTRY', cst.T_EXEC_EXIT: 'EXEC_EXIT', cst.T_PNL: 'PnL_GROSS',
               cst.T_SL_ST: 'SHORT_TERM_SL', cst.T_SL_LT: 'LONG_TERM_SL'}


def ffill_loop(arr, fill=0):

    mask = np.isnan(arr[0])
    arr[0][mask] = fill
    for i in range(1, len(arr)):
        mask = np.isnan(arr[i])
        arr[i][mask] = arr[i - 1][mask]
    return arr


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@njit(fastmath=True)
def compute_trade_pnl(trade, signal_col, entry_price, stop_loss, take_profit,
                      lot_size=200_000, valid=14400):

    orders = np.empty((15, ))
    orders[:] = np.NaN
    orders[cst.T_SL_ST] = 0
    orders[cst.T_SL_LT] = 0

    for i in np.arange(trade.shape[0]):
        signal, close_price = trade[i, signal_col], trade[i, cst.R_BID]
        en, sl, tp = orders[cst.T_ENTRY], orders[cst.T_SL], orders[cst.T_TP]
        order_statut, order_type = orders[cst.T_TRADE_STATUS], orders[cst.T_ORDER_TYPE]

        #################################################################################################
        # Close Buy and Sell Trades that reach target
        #################################################################################################

        if order_statut == cst.T_ACTIVE_TRADE:

            if order_type == cst.T_ORDER_BUY:

                orders[cst.T_PNL] = (close_price - orders[cst.T_EXEC_ENTRY]) * lot_size
                trade_time = trade[i, cst.R_TIMESTAMP] - orders[cst.T_TIMESTAMP]

                if (close_price >= tp) | (close_price <= sl):
                    order_statut = cst.T_CLOSED_TRADE
                    orders[cst.T_TRADE_STATUS] = cst.T_CLOSED_TRADE
                    orders[cst.T_DT_CLOSE] = trade[i, cst.R_TIMESTAMP]
                    orders[cst.T_EXEC_EXIT] = close_price

                    return orders

                if (trade_time < valid) & (orders[cst.T_EXEC_ENTRY] - close_price > orders[cst.T_SL_ST]):
                        orders[cst.T_SL_ST] = orders[cst.T_EXEC_ENTRY] - close_price

                if (trade_time >= valid) & (orders[cst.T_EXEC_ENTRY] - close_price > orders[cst.T_SL_LT]):
                        orders[cst.T_SL_LT] = orders[cst.T_EXEC_ENTRY] - close_price

            elif order_type == cst.T_ORDER_SELL:

                orders[cst.T_PNL] = (orders[cst.T_EXEC_ENTRY] - close_price) * lot_size

                if (close_price <= tp) | (close_price >= sl):
                    order_statut = cst.T_CLOSED_TRADE
                    orders[cst.T_TRADE_STATUS] = cst.T_CLOSED_TRADE
                    orders[cst.T_DT_CLOSE] = trade[i, cst.R_TIMESTAMP]
                    orders[cst.T_EXEC_EXIT] = close_price

                    return orders

                if (trade_time < valid) & (close_price - orders[cst.T_EXEC_ENTRY] > orders[cst.T_SL_ST]):
                    orders[cst.T_SL_ST] = close_price - orders[cst.T_EXEC_ENTRY]
                if (trade_time >= valid) & (close_price - orders[cst.T_EXEC_ENTRY] > orders[cst.T_SL_LT]):
                     orders[cst.T_SL_LT] = close_price - orders[cst.T_EXEC_ENTRY]

        if order_statut == cst.T_PENDING_TRADE:
            if (order_type == cst.T_ORDER_BUY) & (en >= close_price):
                order_statut = cst.T_ACTIVE_TRADE
                orders[cst.T_DT_OPEN] = trade[i, cst.R_TIMESTAMP]
                orders[cst.T_EXEC_ENTRY] = close_price
                orders[cst.T_TRADE_STATUS] = cst.T_ACTIVE_TRADE
            elif (order_type == cst.T_ORDER_SELL) & (en <= close_price):
                order_statut = cst.T_ACTIVE_TRADE
                orders[cst.T_DT_OPEN] = trade[i, cst.R_TIMESTAMP]
                orders[cst.T_EXEC_ENTRY] = close_price
                orders[cst.T_TRADE_STATUS] = cst.T_ACTIVE_TRADE

        if (order_statut == cst.T_PENDING_TRADE) & (trade[i, cst.R_TIMESTAMP] - orders[cst.T_TIMESTAMP] > valid):
            order_statut = cst.T_CANCEL_TRADE
            orders[cst.T_TRADE_STATUS] = cst.T_CANCEL_TRADE
            orders[cst.T_DT_CLOSE] = trade[i, cst.R_TIMESTAMP]
            return orders

        # Create New Orders
        if signal >= 100:
            orders[cst.T_TIMESTAMP] = trade[i, cst.R_TIMESTAMP]  # Signal Time
            orders[ cst.T_ORDER_TYPE] = cst.T_ORDER_BUY  # Order Type
            orders[cst.T_SPREAD] = (trade[i - 1, 7] - trade[i - 1, 8]) * 10_000  # Order Type
            orders[cst.T_PRICE] = close_price  # Close Price
            orders[cst.T_ENTRY] = close_price - entry_price  # Buy Price
            orders[cst.T_SL] = close_price - entry_price - stop_loss  # Stop Loss
            orders[cst.T_TP] = close_price - entry_price + take_profit  # Take Profit
            orders[cst.T_TRADE_STATUS] = cst.T_PENDING_TRADE  # Trade Status

        elif signal <= -100:
            orders[cst.T_TIMESTAMP] = trade[i, cst.R_TIMESTAMP]  # Signal Time
            orders[cst.T_ORDER_TYPE] = cst.T_ORDER_SELL  # Order Type
            orders[cst.T_SPREAD] = (trade[i - 1, 7] - trade[i - 1, 8]) * 10_000  # Order Type
            orders[cst.T_PRICE] = close_price  # Close Price
            orders[cst.T_ENTRY] = close_price + entry_price  # Sell Price
            orders[cst.T_SL] = close_price + entry_price + stop_loss  # Stop Loss
            orders[cst.T_TP] = close_price + entry_price - take_profit  # Take Profit
            orders[cst.T_TRADE_STATUS] = cst.T_PENDING_TRADE  # Trade status

    orders[cst.T_DT_CLOSE] = trade[-1, cst.R_TIMESTAMP]

    return orders


class FastHighProbCandleStick:

    params = {'ind': 'CDLENGULFING', 'entry_pip': 0, 'exit_pip': 5, 'stop_loss': 45, 'max_spread': 2}

    HIGH_PROBABILITIES_CANDLES = [
        'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDLADVANCEBLOCK', 'CDLBELTHOLD',
        'CDLCLOSINGMARUBOZU', 'CDLDARKCLOUDCOVER', 'CDLDOJISTAR', 'CDLENGULFING',
        'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHANGINGMAN',
        'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
        'CDLLONGLINE', 'CDLMARUBOZU', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLXSIDEGAP3METHODS'
    ]

    FREQ = ['1MIN', '5MIN', '15MIN', '1H', '4H', '1D']
    RE_SAMPLE_FREQ = {'1MIN': cst.TICK_1MIN, '5MIN': cst.TICK_5MIN, '15MIN': cst.TICK_15MIN,
                      '1H': cst.TICK_60MIN, '4H': cst.TICK_4H, '1D': cst.TICK_1D}

    TOHLC = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
    MAPPING = {'15MIN': [2, 13, 35, 14, 15, 16, 17, 41]}

    def initialize_labels(self):

        label_data = {'TICK': 0, 'TIMESTAMP': 1, 'BID': 2, 'ASK': 3}
        label_data_r = {0: 'TICK', 1: 'TIMESTAMP', 2: 'BID', 3: 'ASK'}

        for i in range(len(self.re_sample_frequency)):
            j = 4 + i * 6
            for name in ['BAR_{}', 'NEW_BAR_{}', 'OPEN_{}', 'HIGH_{}', 'LOW_{}', 'CLOSE_{}']:
                label_data[name.format(self.re_sample_frequency[i])] = j
                label_data_r[j] = name.format(self.re_sample_frequency[i])
                j += 1
        for freq in self.candle_frequency:
            for candle in self.candle_frequency[freq]:
                label_data[candle + f'_{freq}'] = j
                label_data_r[j] = candle + f'_{freq}'
                j += 1
        #
        # for freq in self.other_signals:
        #     for candle in self.other_signals[freq]:
        #         label_data[candle + f'_{freq}'] = j
        #         label_data_r[j] = candle + f'_{freq}'
        #         j += 1

        self.label_data = label_data
        self.label_data_r = label_data_r

    @staticmethod
    @njit()
    def re_sample_data(df, n_frequency, n_columns, re_sample_columns=2):

        resample = np.empty((df.shape[0], n_columns), dtype=np.float64)
        resample[:] = np.NaN
        resample[:, cst.R_TIMESTAMP:cst.R_ASK + 1] = df

        resample[:, n_frequency[:, 1] + cst.R_NEW_BAR] = 0
        ohlc = np.empty((n_frequency.shape[0], 4), dtype=np.float64)

        for i in prange(0, resample.shape[0]):
            resample[i, cst.R_TICK] = i + 1
            for k in np.arange(n_frequency.shape[0]):
                if n_frequency[k, 0] == cst.TICK_1MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_1MIN
                if n_frequency[k, 0] == cst.TICK_5MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_5MIN
                if n_frequency[k, 0] == cst.TICK_15MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_15MIN
                if n_frequency[k, 0] == cst.TICK_60MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_60MIN
                if n_frequency[k, 0] == cst.TICK_4H:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_4H
                if n_frequency[k, 0] == cst.TICK_1D:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) // cst.TICK_1D

            # Initialisation
            if i == 0:
                for k in np.arange(n_frequency.shape[0]):
                    resample[i,  n_frequency[k, 1] + cst.R_OPEN: n_frequency[k, 1] + cst.R_CLOSE + 1] = resample[i, re_sample_columns]
                ohlc[:, :] = resample[i, re_sample_columns]
            else:
                for j in prange(n_frequency.shape[0]):
                    if resample[i, 6 * j + 4] != resample[i - 1, 6 * j + 4]:
                        ohlc[j, 0] = resample[i, re_sample_columns]
                        ohlc[j, 1] = resample[i, re_sample_columns]
                        ohlc[j, 2] = resample[i, re_sample_columns]
                        ohlc[j, 3] = resample[i, re_sample_columns]
                        resample[i, n_frequency[j, 1] + cst.R_NEW_BAR] = 1
                        resample[i - 1, n_frequency[j, 1] + cst.R_NEW_BAR] = -1

                ohlc[ohlc[:, 1] < resample[i, re_sample_columns], 1] = resample[i, re_sample_columns]
                ohlc[ohlc[:, 2] > resample[i, re_sample_columns], 2] = resample[i, re_sample_columns]
                ohlc[:, 3] = resample[i, re_sample_columns]

                for k in np.arange(n_frequency.shape[0]):
                    resample[i,  n_frequency[k, 1] + cst.R_OPEN: n_frequency[k, 1] + cst.R_CLOSE + 1] = ohlc[k]

        return resample

    def __init__(self, fx_data_dict, price_for_signal, entry_pip, stop_loss, take_profit, valid, lot_size,
                 re_sample_frequency, signals, other_signals):

        self.fx_manager = {}
        self.__entry_pip = entry_pip
        self.__stop_loss = stop_loss
        self.__take_profit = take_profit
        self.__lot_size = lot_size
        self.__valid = valid
        self.re_sample_frequency = re_sample_frequency
        self.candle_frequency = signals
        self.other_signals = other_signals
        self.other_signals_cols = {}
        self.initialize_labels()
        to_resample = np.array([[self.RE_SAMPLE_FREQ[i], 1] for i in self.re_sample_frequency])
        to_resample[:, 1] = 4 + (np.cumsum(to_resample[:, 1]) - 1) * 6
        print(self.label_data)

        for fx_pair in fx_data_dict:
            self.fx_manager[fx_pair] = self.re_sample_data(
                fx_data_dict[fx_pair], n_frequency=to_resample, n_columns=len(self.label_data) + 73,
                re_sample_columns=price_for_signal
            )

            self.add_candlestick_pattern(fx_pair)
            self.add_other_signals(fx_pair)

    def add_candlestick_pattern(self, fx_pair):
        """
        Description:
        ------------

        Add Candle stick pattern to the price.
        """

        df = self.fx_manager[fx_pair]
        to_fill = []
        for freq in self.candle_frequency:
            open_pos = self.label_data[f'OPEN_{freq}']
            high_pos = self.label_data[f'HIGH_{freq}']
            low_pos = self.label_data[f'LOW_{freq}']
            close_pos = self.label_data[f'CLOSE_{freq}']
            new_bar_pos = self.label_data[f'NEW_BAR_{freq}']

            _df = df[df[:, new_bar_pos] == -1]
            for candle in self.candle_frequency[freq]:
                pattern = getattr(talib, candle)(_df[:, open_pos], _df[:, high_pos], _df[:, low_pos], _df[:, close_pos])
                pos = self.label_data[candle + f'_{freq}']
                to_fill.append(pos)
                self.fx_manager[fx_pair][:, pos] = 0
                self.fx_manager[fx_pair][_df[:, 0].astype(np.int64), pos] = pattern

        return df

    @staticmethod
    def __add_technical_indicators(df, frequency):

        # add_momentum_ta, add_trend_ta, add_volatility_ta

        df = add_momentum_ta(
            df, high=cst.R_HIGH, low=cst.R_LOW, close=cst.R_CLOSE, volume=cst.R_VOLUME,
            colprefix=frequency + '_')

        df = add_trend_ta(
            df, high=cst.R_HIGH, low=cst.R_LOW, close=cst.R_CLOSE,
            colprefix=frequency + '_'
        )

        df = add_volatility_ta(
            df, high=cst.R_HIGH, low=cst.R_LOW, close=cst.R_CLOSE,
            colprefix=frequency + '_'
        )

        return df

    def add_other_signals(self, fx_pair):
        """
        Description:
        ------------

        Add Candle stick pattern to the price.
        """

        df = self.fx_manager[fx_pair]
        to_fill = []
        for freq in self.other_signals:
            open_pos = self.label_data[f'OPEN_{freq}']
            high_pos = self.label_data[f'HIGH_{freq}']
            low_pos = self.label_data[f'LOW_{freq}']
            close_pos = self.label_data[f'CLOSE_{freq}']
            new_bar_pos = self.label_data[f'NEW_BAR_{freq}']

            _df = df[df[:, new_bar_pos] == -1, ]

            _df = pd.DataFrame(
                _df[:, [high_pos, low_pos, close_pos]],
                columns=[cst.R_HIGH, cst.R_LOW, cst.R_CLOSE],
                index=_df[:, cst.R_TICK]
            )
            _df[cst.R_VOLUME] = np.NaN

            _df = self.__add_technical_indicators(_df, freq).iloc[:, 4:]
            i = len(self.label_data)
            for cols in _df.columns:
                if cols not in self.label_data:
                    self.label_data[cols] = i
                    self.label_data_r[i] = cols
                    self.other_signals_cols[cols] = i
                    i += 1
            _df.rename(columns=self.label_data, inplace=True)
            self.fx_manager[fx_pair][np.ix_(_df.index.astype(np.int64), _df.columns.tolist())] = _df.values
            continue
            for signal in self.other_signals[freq]:
                if signal in ['Z_ADX']:
                    pattern = getattr(talib, 'ADX')(high=_df[:, high_pos], low=_df[:, low_pos], close=_df[:, close_pos])
                    mean = np.mean(rolling_window(pattern, 14), 1)
                    std = np.std(rolling_window(pattern, 14), 1)
                    value = (pattern[13:] - mean) / std

                    pos = self.label_data[signal + f'_{freq}']
                    self.fx_manager[fx_pair][:, pos] = np.NaN
                    self.fx_manager[fx_pair][_df[13:, 0].astype(np.int64), pos] = value
                    to_fill.append(pos)
                elif signal in ['DIR']:
                    minus = getattr(talib, 'MINUS_DI')(high=_df[:, high_pos], low=_df[:, low_pos], close=_df[:, close_pos])
                    plus = getattr(talib, 'PLUS_DI')(high=_df[:, high_pos], low=_df[:, low_pos], close=_df[:, close_pos])

                    value = plus - minus
                    mean = np.mean(rolling_window(value, 14), 1)
                    std = np.std(rolling_window(value, 14), 1)
                    value = (value[13:] - mean) / std

                    pos = self.label_data[signal + f'_{freq}']
                    self.fx_manager[fx_pair][:, pos] = np.NaN
                    self.fx_manager[fx_pair][_df[13:, 0].astype(np.int64), pos] = value
                    to_fill.append(pos)

                    pass

        # to_fill = _df.columns.tolist()
        # self.fx_manager[fx_pair][:, to_fill] = ffill_loop(self.fx_manager[fx_pair][:, to_fill], fill=np.NaN)
        # print(self.fx_manager[fx_pair])
        # return df

    @staticmethod
    @njit()
    def run_serial_back_test(df, signal_col, entry_price, stop_loss, take_profit, lot_size=200_000, valid=14400):

        orders = np.empty((df[df[:, signal_col] != 0].shape[0], 17))
        orders[:, cst.T_SL_ST] = 0
        orders[:, cst.T_SL_LT] = 0

        order_number = 1
        
        for i in np.arange(df.shape[0]):
            signal = df[i, signal_col]
            close_price = df[i, cst.R_BID]

            en = orders[:, cst.T_ENTRY]
            sl = orders[:, cst.T_SL]
            tp = orders[:, cst.T_TP]

            # Close Buy and Sell Trades that reach target

            # Close Close Take Profit Buy Trade
            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_ACTIVE_TRADE) & ((close_price >= tp) | (close_price <= sl))
            mask &= (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_BUY)
            orders[mask, cst.T_TRADE_STATUS] = cst.T_CLOSED_TRADE
            orders[mask, cst.T_DT_CLOSE] = df[i, cst.R_TIMESTAMP]
            orders[mask, cst.T_EXEC_EXIT] = close_price
            orders[mask, cst.T_PNL] = (close_price - orders[mask, cst.T_EXEC_ENTRY]) * lot_size

            # Close Close Take Profit Sell Trade
            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_ACTIVE_TRADE) & ((close_price <= tp) | (close_price >= sl))
            mask &= (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_SELL)
            orders[mask, cst.T_TRADE_STATUS] = cst.T_CLOSED_TRADE
            orders[mask, cst.T_DT_CLOSE] = df[i, cst.R_TIMESTAMP]
            orders[mask, cst.T_EXEC_EXIT] = close_price
            orders[mask, cst.T_PNL] = (orders[mask, cst.T_EXEC_ENTRY] - close_price) * lot_size

            # Buy Order Execute
            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_PENDING_TRADE)  & (en >= close_price)
            mask &= (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_BUY)
            orders[mask, cst.T_DT_OPEN] = df[i, cst.R_TIMESTAMP]
            orders[mask, cst.T_EXEC_ENTRY] = close_price
            orders[mask, cst.T_TRADE_STATUS] = cst.T_ACTIVE_TRADE

            # SELL Order Execute
            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_PENDING_TRADE) & (en <= close_price)
            mask &= (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_SELL)
            orders[mask, cst.T_DT_OPEN] = df[i, cst.R_TIMESTAMP]
            orders[mask, cst.T_EXEC_ENTRY] = close_price
            orders[mask, cst.T_TRADE_STATUS] = cst.T_ACTIVE_TRADE

            # Cancel Order if invalid
            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_PENDING_TRADE)
            mask &= (df[i, cst.R_TIMESTAMP] - orders[:, cst.T_TIMESTAMP] > valid)
            orders[mask, cst.T_TRADE_STATUS] = cst.T_CANCEL_TRADE
            orders[mask, cst.T_DT_CLOSE] = df[i, cst.R_TIMESTAMP]
            
            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_ACTIVE_TRADE) & (df[i, cst.R_TIMESTAMP] - orders[:, cst.T_TIMESTAMP] < valid)
            m1 = mask & (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_BUY) & (orders[:, cst.T_EXEC_ENTRY] - close_price > orders[:, cst.T_SL_ST])
            orders[m1, cst.T_SL_ST] = orders[m1, cst.T_EXEC_ENTRY] - close_price
            m1 = mask & (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_SELL) & (close_price - orders[:, cst.T_EXEC_ENTRY] > orders[:, cst.T_SL_ST])
            orders[m1, cst.T_SL_ST] = close_price - orders[m1, cst.T_EXEC_ENTRY]

            mask = (orders[:, cst.T_TRADE_STATUS] == cst.T_ACTIVE_TRADE) & (df[i, cst.R_TIMESTAMP] - orders[:, cst.T_TIMESTAMP] >= valid)
            m1 = mask & (orders[:, cst.T_ORDER_TYPE] == cst.T_ORDER_BUY) & (orders[:, cst.T_EXEC_ENTRY] - close_price > orders[:, cst.T_SL_LT])
            orders[m1, cst.T_SL_LT] = orders[m1, cst.T_EXEC_ENTRY] - close_price
            m1 = mask & (orders[:,cst.T_ORDER_TYPE] == cst.T_ORDER_SELL) & (close_price - orders[:, cst.T_EXEC_ENTRY] > orders[:, cst.T_SL_LT])
            orders[m1, cst.T_SL_LT] = close_price - orders[m1, cst.T_EXEC_ENTRY]

            # Create New Orders
            if signal >= 100:
                orders[order_number - 1, cst.T_TIMESTAMP] = df[i, cst.R_TIMESTAMP]       # Signal Time
                orders[order_number - 1, cst.T_ORDER_TYPE] = cst.T_ORDER_BUY              # Order Type
                orders[order_number - 1, cst.T_SPREAD] = (df[i - 1, 7] - df[i - 1, 8]) * 10_000             # Order Type
                orders[order_number - 1, cst.T_PRICE] = close_price    # Close Price
                orders[order_number - 1, cst.T_ENTRY] = close_price - entry_price   # Buy Price
                orders[order_number - 1, cst.T_SL] = close_price - entry_price - stop_loss   # Stop Loss
                orders[order_number - 1, cst.T_TP] = close_price - entry_price + take_profit   # Take Profit
                orders[order_number - 1, cst.T_TRADE_STATUS] = cst.T_PENDING_TRADE   # Trade Status

                orders[order_number - 1, -2:] = df[i, -2:]       # Signal Time

                order_number += 1
            elif signal <= -100:
                orders[order_number - 1, cst.T_TIMESTAMP] = df[i, cst.R_TIMESTAMP]  # Signal Time
                orders[order_number - 1, cst.T_ORDER_TYPE] = cst.T_ORDER_SELL  # Order Type
                orders[order_number - 1, cst.T_SPREAD] = (df[i - 1, 7] - df[i - 1, 8]) * 10_000             # Order Type
                orders[order_number - 1, cst.T_PRICE] = close_price  # Close Price
                orders[order_number - 1, cst.T_ENTRY] = close_price + entry_price  # Sell Price
                orders[order_number - 1, cst.T_SL] = close_price + entry_price + stop_loss  # Stop Loss
                orders[order_number - 1, cst.T_TP] = close_price + entry_price - take_profit  # Take Profit
                orders[order_number - 1, cst.T_TRADE_STATUS] = cst.T_PENDING_TRADE  # Trade status

                orders[order_number - 1, -2:] = df[i, -2:]

                order_number += 1

        return orders

    @staticmethod
    # @njit(parallel=True, fastmath=True)
    def run_parallel_back_test(df, signal_cols, other_signal_cols, entry_price, stop_loss, take_profit,
                               lot_size=200_000, valid=14400):
        n_trades = 0
        candle_start = np.empty(signal_cols.shape, dtype=np.int64)

        for j in np.arange(signal_cols.shape[0]):
            candle_start[j] = n_trades
            n_trades += df[(df[:, signal_cols[j]] >= 100) | (df[:, signal_cols[j]] <= -100)].shape[0]
        orders = np.empty((n_trades, 16 + other_signal_cols.shape[0]))

        for signal_pos in prange(signal_cols.shape[0]):
            signal_col = signal_cols[signal_pos]
            all_trades = df[(df[:, signal_col] >= 100) | (df[:, signal_col] <= -100)]
            j = candle_start[signal_pos]

            for i in prange(all_trades.shape[0]):

                trade = all_trades[i]
                ts, tick = trade[cst.R_TIMESTAMP], trade[cst.R_TICK]
                end_ts = ts + 5 * 3600 * 24
                period_trade = df[(df[:, cst.R_TIMESTAMP] >= ts - 100) & (df[:, cst.R_TIMESTAMP] <= end_ts)]
                period_trade[period_trade[:, cst.R_TICK] != tick, signal_col] = 0
                orders[i + j, :15] = compute_trade_pnl(
                    trade=period_trade, signal_col=signal_col, entry_price=entry_price, stop_loss=stop_loss,
                    take_profit=take_profit, lot_size=lot_size, valid=valid
                )
                orders[i + j, 15 + other_signal_cols - np.min(other_signal_cols)] = all_trades[i, other_signal_cols]
                orders[i + j, -1] = signal_col
        return orders

    def run(self):

        results = []
        min_other_signal = min(self.other_signals_cols.values())
        for i in self.other_signals_cols:
            RESULT_COLS[15 + self.other_signals_cols[i] - min_other_signal] = i

        for fx_pair in self.fx_manager:

            df = self.fx_manager[fx_pair]
            all_pos = np.array([self.label_data[candle + f'_{freq}'] for freq in self.candle_frequency
                                for candle in self.candle_frequency[freq]])

            r = self.run_parallel_back_test(
                df, signal_cols=all_pos, entry_price=self.__entry_pip / cst.FX_PIP[fx_pair],
                stop_loss=self.__stop_loss / cst.FX_PIP[fx_pair],
                take_profit=self.__take_profit / cst.FX_PIP[fx_pair],
                lot_size=self.__lot_size, valid=self.__valid,
                other_signal_cols=np.array(list(self.other_signals_cols.values()))
            )
            r = pd.DataFrame(r, ).rename(columns=RESULT_COLS)

            candle_cols = r.iloc[:, -1].replace(self.label_data_r).str.split('_', expand=True)
            r['DT_TIME'] = r['DT_CLOSE'] - r['DT_OPEN']
            r['TIMESTAMP'] = (r['TIMESTAMP'] * 10 ** 9).astype('datetime64[ns]')
            r['DT_OPEN'] = (r['DT_OPEN'] * 10 ** 9).astype('datetime64[ns]')
            r['DT_CLOSE'] = (r['DT_CLOSE'] * 10 ** 9).astype('datetime64[ns]')
            r['SHORT_TERM_SL'] *= cst.FX_PIP[fx_pair]
            r['LONG_TERM_SL'] *= cst.FX_PIP[fx_pair]

            r['CANDLE'] = candle_cols[0]
            r['SIGNAL_FREQ'] = candle_cols[1]
            r['FX_PAIR'] = fx_pair
            results.append(r)
            print(f'Done for {fx_pair}')

            # for freq in self.candle_frequency:
            #
            #     for candle in self.candle_frequency[freq]:
            #         pos = self.label_data[candle + f'_{freq}']
            #         r = self.run_parallel_back_test(
            #             df, signal_col=pos, entry_price=self.__entry_pip / cst.FX_PIP[fx_pair],
            #             stop_loss=self.__stop_loss / cst.FX_PIP[fx_pair],
            #             take_profit=self.__take_profit / cst.FX_PIP[fx_pair],
            #             lot_size=self.__lot_size, valid=self.__valid
            #         )
            #         r = pd.DataFrame(r, ).rename(columns=RESULT_COLS)
            #
            #         r['DT_TIME'] = r['DT_CLOSE'] - r['DT_OPEN']
            #         r['TIMESTAMP'] = (r['TIMESTAMP'] * 10 ** 9).astype('datetime64[ns]')
            #         r['DT_OPEN'] = (r['DT_OPEN'] * 10 ** 9).astype('datetime64[ns]')
            #         r['DT_CLOSE'] = (r['DT_CLOSE'] * 10 ** 9).astype('datetime64[ns]')
            #         r['SHORT_TERM_SL'] *= cst.FX_PIP[fx_pair]
            #         r['LONG_TERM_SL'] *= cst.FX_PIP[fx_pair]
            #         r['CANDLE'] = candle
            #         r['SIGNAL_FREQ'] = freq
            #         r['FX_PAIR'] = fx_pair
            #         print(r.head(10))
            #         return
            #         results.append(r)
            #         print(f'Done for {fx_pair}, Candle: {candle}')

        return pd.concat(results)


def parse_args(pargs=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Back-test High probability candlesticks patterns')

    parser.add_argument('--fx_pair', '-fx', required=False, default='eurgbp',
                        choices=list(cst.FX_PIP.keys()), help='FX pair to backtest')
    parser.add_argument('--backtest_column', '-target_columns', required=False, default=cst.R_BID,
                        choices=[cst.R_BID, cst.R_ASK], help='Columns used for the back test.')
    # parser.add_argument('--year', '-y', required=False, default=[2022])

    parser.add_argument('--entry_pip', '-entry', required=False, default=0)
    parser.add_argument('--take_profit', '-tp', required=False, default=5)
    parser.add_argument('--stop_loss', '-sl', required=False, default=45)
    parser.add_argument('--valid', '-vld', required=False, default=14400)
    parser.add_argument('--lot_size', '-ls', required=False, default=200_000)

    parser.add_argument('--indicator', required=False, action='store', default='CDLENGULFING',
                        choices=FastHighProbCandleStick.HIGH_PROBABILITIES_CANDLES,
                        help=('Which indicator pair to show together'))

    return parser.parse_args()


def run_strategy(pair, args=None):

    args = parse_args(args)
    data = pd.read_parquet(RAW_DATA_PATH + '/sample.parquet')

    # data = read_data(cst.FX_TICK_CHUNK_DATA_PATH, fx_pair=[pair], date_range=[2022])
    data['TIMESTAMP'] = data.index.astype(np.int64) // 10 ** 9
    cols = ['TIMESTAMP', 'BID', 'ASK']

    signals = {'15MIN': ['CDLENGULFING']}
    other_signals = {'5MIN': ['Z_ADX', 'DIR']}

    # signals = {'15MIN': talib.get_function_groups()['Pattern Recognition'], }
    t = time()
    bt = FastHighProbCandleStick(
        fx_data_dict={pair: data[cols].values}, entry_pip=args.entry_pip, stop_loss=args.stop_loss,
        take_profit=args.take_profit, price_for_signal=args.backtest_column, valid=args.valid,
        lot_size=args.lot_size, signals=signals, re_sample_frequency=['15MIN', '5MIN'],
        other_signals=other_signals
    )
    strategy_result = bt.run()
    print(time() - t)
    print(strategy_result)
    strategy_result.to_csv('test2.csv')
    del bt
    del data
    return strategy_result
    strategy_result = pd.read_csv('all_pair.csv')
    # strategy_result['TIMESTAMP'] = strategy_result['TIMESTAMP'].astype('datetime64[ns]')
    # strategy_result['DT_CLOSE'] = strategy_result['DT_CLOSE'].astype('datetime64[ns]')
    #
    # #
    # threshold = [10, 15, 20, 25, 30, 35, 40, 45]
    # by = {
    #     'CANDLE': {i: i for i in talib.get_function_groups()['Pattern Recognition']},
    #     'ORDER_TYPE': {-1: 'BEARISH', 1: 'BULLISH'}, 'FX_PAIR': {'eurgbp': 'EUR/GBP'},
    #     'SIGNAL_HOUR': {0: '0H-4H', 1: '4H-8H', 2: '8H-12H', 3: '12H-16H', 4: '16H-20H', 5: '20H-24H', },
    #     'SIGNAL_CLOSE': {0: '0H-4H', 1: '4H-8H', 2: '8H-12H', 3: '12H-16H', 4: '16H-20H', 5: '20H-24H', },
    #     'SPREAD_SIZE': {**{0: '0-10PIPS', 1: '10-20PIPS', 2: '20-30PIPS'}, **{i: '30PIPS+' for i in range(3, 10)}},
    #     'TIME_CHUNK': {**{0: '0H-4H', 1: '4H-8H', 2: '8H-12H', 3: '12H-16H', 4: '16H-20H', 5: '20H-24H', },
    #                    **{i: '24H+' for i in range(6, 100)}}
    #
    # }
    # stats = StrategyStats(df=strategy_result, by=by, threshold=threshold)
    # mask = (strategy_result['SIGNAL_HOUR'].isin([2, 3, 4]))
    # strategy_result = strategy_result[mask]
    # print(stats.compute_trade(strategy_result, by))
    #
    # return args


class StrategyStats:

    def __init__(self, df, by, threshold):

        self.__df = df
        self.__by = by
        self.__threshold = threshold

        self.__transform_data()
        pass

    def __transform_data(self):

        self.__df['SIGNAL_HOUR'] = (self.__df['TIMESTAMP'].dt.hour / 4).astype(int)
        self.__df['SIGNAL_CLOSE'] = (self.__df['DT_CLOSE'].dt.hour / 4).astype(int)
        self.__df['SPREAD_SIZE'] = (self.__df['SPREAD'] / 10).astype(int)
        self.__df['TIME_CHUNK'] = (self.__df['DT_TIME'] / (4 * 60 * 60)).astype(int)

    def compute_pnl(self, df, by):

        r = {}

        for i in self.__threshold:
            pnl_df = df.copy()
            mask = (pnl_df['SHORT_TERM_SL'] >= i) | (pnl_df['LONG_TERM_SL'] >= i)
            pnl_df.loc[mask, 'PnL_GROSS'] = -20 * i
            group_stats  = []
            for group in by:
                pnl_df[group] = pnl_df[group].replace(by[group])
                _r = {}
                _r[(group, 'Winning')] = pnl_df[pnl_df["PnL_GROSS"] > 0].groupby(group)['PnL_GROSS'].sum()
                _r[(group, 'Losing')] = pnl_df[pnl_df["PnL_GROSS"] < 0].groupby(group)['PnL_GROSS'].sum()
                _r[(group, 'Total')] = pnl_df.groupby(group)['PnL_GROSS'].sum()
                _r = pd.concat(_r)
                _r.index.names = ['BY', 'DETAIL', 'KEY_NAME']
                group_stats.append(_r)
            r[i] = pd.concat(group_stats)
        return pd.concat(r, axis=1)

    def compute_trade(self, df, by):

        r = {}

        for i in self.__threshold:
            trade_df = df.copy()
            mask = (trade_df['SHORT_TERM_SL'] >= i) | (trade_df['LONG_TERM_SL'] >= i)
            trade_df.loc[mask, 'PnL_GROSS'] = -20 * i
            group_stats = []
            for group in by:
                trade_df[group] = trade_df[group].replace(by[group])
                _r = {}
                _r[(group, 'Winning')] = trade_df[trade_df["PnL_GROSS"] > 0].groupby(group)['PnL_GROSS'].count()
                _r[(group, 'Losing')] = trade_df[trade_df["PnL_GROSS"] < 0].groupby(group)['PnL_GROSS'].count()
                _r[(group, 'Total')] = trade_df.groupby(group)['PnL_GROSS'].count()
                _r = pd.concat(_r)
                _r.index.names = ['BY', 'DETAIL', 'KEY_NAME']
                group_stats.append(_r)
            r[i] = pd.concat(group_stats)
        return pd.concat(r, axis=1)

    def compute_win_ratio(self, df, by):

        r = {}

        for i in self.__threshold:
            wr_df = df.copy()
            mask = (wr_df['SHORT_TERM_SL'] >= i) | (wr_df['LONG_TERM_SL'] >= i)
            wr_df.loc[mask, 'PnL_GROSS'] = -20 * i
            group_stats = []
            for group in by:
                _r = {}
                wr_df[group] = wr_df[group].replace(by[group])
                winning = wr_df[wr_df["PnL_GROSS"] > 0].groupby(group)['PnL_GROSS'].count()
                total = wr_df.groupby(group)['PnL_GROSS'].count()
                _r[(group, 'Winning')] = 100 * winning / total
                _r = pd.concat(_r)
                _r.index.names = ['BY', 'DETAIL', 'KEY_NAME']
                group_stats.append(_r)
            r[i] = pd.concat(group_stats)
        return pd.concat(r, axis=1).round(1)

    def compute_risk_reward(self, df, by):
        r = {}

        for i in self.__threshold:
            rr_df = df.copy()
            mask = (rr_df['SHORT_TERM_SL'] >= i) | (rr_df['LONG_TERM_SL'] >= i)
            rr_df.loc[mask, 'PnL_GROSS'] = -20 * i
            group_stats = []
            for group in by:
                _r = {}
                rr_df[group] = rr_df[group].replace(by[group])
                winning = rr_df[rr_df["PnL_GROSS"] > 0].groupby(group)['PnL_GROSS'].mean()
                losing = -rr_df[rr_df["PnL_GROSS"] < 0].groupby(group)['PnL_GROSS'].mean()
                _r[(group, 'Winning')] =  winning / losing
                _r = pd.concat(_r)
                _r.index.names = ['BY', 'DETAIL', 'KEY_NAME']
                group_stats.append(_r)
            r[i] = pd.concat(group_stats)
        return pd.concat(r, axis=1).round(3)

    def compute_expectation(self, df, by):

        r = {}

        for i in self.__threshold:
            ep_df = df.copy()
            mask = (ep_df['SHORT_TERM_SL'] >= i) | (ep_df['LONG_TERM_SL'] >= i)
            ep_df.loc[mask, 'PnL_GROSS'] = -20 * i
            group_stats = []
            for group in by:
                _r = {}
                ep_df[group] = ep_df[group].replace(by[group])
                win_ratio = ep_df[ep_df["PnL_GROSS"] > 0].groupby(group)['PnL_GROSS'].count()
                total = ep_df.groupby(group)['PnL_GROSS'].count()
                win_ratio /= total
                winning = ep_df[ep_df["PnL_GROSS"] > 0].groupby(group)['PnL_GROSS'].mean()
                losing = ep_df[ep_df["PnL_GROSS"] < 0].groupby(group)['PnL_GROSS'].mean()

                expectation = 0
                if len(winning) > 0:
                    expectation += winning * win_ratio
                if len(losing) > 0:
                    expectation += (1-win_ratio) * losing

                _r[(group, 'Winning')] = expectation
                _r = pd.concat(_r)
                _r.index.names = ['BY', 'DETAIL', 'KEY_NAME']
                group_stats.append(_r)
            r[i] = pd.concat(group_stats)
        return pd.concat(r, axis=1).round(3)

    def compute_maximum_drawdown(self, df, by):

        r = {}

        for i in self.__threshold:
            ep_df = df.copy().sort_values(by=['DT_CLOSE'])
            mask = (ep_df['SHORT_TERM_SL'] >= i) | (ep_df['LONG_TERM_SL'] >= i)
            ep_df.loc[mask, 'PnL_GROSS'] = -20 * i
            group_stats = []
            for group in by:
                _r = {}
                ep_df[group] = ep_df[group].replace(by[group])
                dd = ep_df.groupby(group)['PnL_GROSS'].apply(lambda x: (x.cumsum() - x.cumsum().cummax()).min())
                _r[(group, 'Winning')] = dd
                _r = pd.concat(_r)
                _r.index.names = ['BY', 'DETAIL', 'KEY_NAME']
                group_stats.append(_r)
            r[i] = pd.concat(group_stats)
        return pd.concat(r, axis=1).round(3)

    def compute_volatility(self):
        pass

    def compute_sharpe_ratio(self):
        pass


if __name__ == '__main__':
    a = []
    for pair in list(cst.BACK_TEST.keys())[10:11]:
        # run_strategy(pair).to_csv(f'{pair}_extended.csv')
        run_strategy(pair)
