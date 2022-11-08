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

RE_SAMPLE_FREQ = {'1MIN': cst.TICK_1MIN, '5MIN': cst.TICK_5MIN, '15MIN': cst.TICK_15MIN,
                  '1H': cst.TICK_60MIN, '4H': cst.TICK_4H, '1D': cst.TICK_1D}

RESULT_COLS = {cst.T_TIMESTAMP: 'TIMESTAMP', cst.T_ORDER_TYPE: 'ORDER_TYPE', cst.T_SPREAD: 'SPREAD',
               cst.T_PRICE: 'PRICE', cst.T_ENTRY: 'ENTRY', cst.T_SL: 'SL', cst.T_TP: 'TP',
               cst.T_TRADE_STATUS: 'TRADE_STATUS', cst.T_DT_OPEN: 'DT_OPEN', cst.T_DT_CLOSE: 'DT_CLOSE',
               cst.T_EXEC_ENTRY: 'EXEC_ENTRY', cst.T_EXEC_EXIT: 'EXEC_EXIT', cst.T_PNL: 'PnL_GROSS',
               cst.T_SL_ST: 'SHORT_TERM_SL', cst.T_SL_LT: 'LONG_TERM_SL'}


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
                trade_time = trade[i, cst.R_TIMESTAMP] - orders[cst.T_TIMESTAMP]

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


class DataReSample:

    def initialize_labels(self):

        label_data = {'TICK': 0, 'TIMESTAMP': 1, 'BID': 2, 'ASK': 3}
        label_data_r = {0: 'TICK', 1: 'TIMESTAMP', 2: 'BID', 3: 'ASK'}

        for i in range(len(self.re_sample_frequency)):
            j = 4 + i * 6
            for name in ['BAR_{}', 'NEW_BAR_{}', 'OPEN_{}', 'HIGH_{}', 'LOW_{}', 'CLOSE_{}']:
                label_data[name.format(self.re_sample_frequency[i])] = j
                label_data_r[j] = name.format(self.re_sample_frequency[i])
                j += 1
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
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[
                                                                      i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_1MIN
                if n_frequency[k, 0] == cst.TICK_5MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[
                                                                      i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_5MIN
                if n_frequency[k, 0] == cst.TICK_15MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[
                                                                      i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_15MIN
                if n_frequency[k, 0] == cst.TICK_60MIN:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[
                                                                      i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_60MIN
                if n_frequency[k, 0] == cst.TICK_4H:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[
                                                                      i, cst.R_TIMESTAMP] - 1) % 86400 // cst.TICK_4H
                if n_frequency[k, 0] == cst.TICK_1D:
                    resample[i, n_frequency[k, 1] + cst.R_BAR] = (resample[i, cst.R_TIMESTAMP] - 1) // cst.TICK_1D

            # Initialisation
            if i == 0:
                for k in np.arange(n_frequency.shape[0]):
                    resample[i, n_frequency[k, 1] + cst.R_OPEN: n_frequency[k, 1] + cst.R_CLOSE + 1] = resample[
                        i, re_sample_columns]
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
                    resample[i, n_frequency[k, 1] + cst.R_OPEN: n_frequency[k, 1] + cst.R_CLOSE + 1] = ohlc[k]

        return resample

    def __init__(self, fx_data_dict, re_sample_frequency, price_for_signal, n_signals):
        self.fx_manager = {}
        self.re_sample_frequency = re_sample_frequency
        self.initialize_labels()
        to_resample = np.array([[RE_SAMPLE_FREQ[i], 1] for i in self.re_sample_frequency])
        to_resample[:, 1] = 4 + (np.cumsum(to_resample[:, 1]) - 1) * 6

        for fx_pair in fx_data_dict:
            self.fx_manager[fx_pair] = self.re_sample_data(
                fx_data_dict[fx_pair], n_frequency=to_resample, n_columns=len(self.label_data) + n_signals,
                re_sample_columns=price_for_signal
            )


class FastBackTesting(DataReSample):

    def __init__(self, fx_data_dict, re_sample_frequency, price_for_signal, n_signals,
                 entry_pip, stop_loss, take_profit, valid, lot_size):

        super().__init__(
            fx_data_dict=fx_data_dict, re_sample_frequency=re_sample_frequency,
            price_for_signal=price_for_signal, n_signals=n_signals
        )
        self.__entry_pip = entry_pip
        self.__stop_loss = stop_loss
        self.__take_profit = take_profit
        self.__lot_size = lot_size
        self.__valid = valid
        self.other_signals_cols = {}

    @staticmethod
    @njit(fastmath=True)
    def run_parallel_back_test(df, signal_cols, entry_price, stop_loss,
                               take_profit, lot_size=200_000, valid=14400):
        n_trades = 0
        candle_start = np.empty(signal_cols.shape, dtype=np.int64)

        for j in np.arange(signal_cols.shape[0]):
            candle_start[j] = n_trades
            n_trades += df[(df[:, signal_cols[j]] >= 100) | (df[:, signal_cols[j]] <= -100)].shape[0]
        orders = np.empty((n_trades, 16))

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
                orders[i + j, -1] = signal_col
        return orders

    def run(self):

        results = []
        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            print(self.label_data)
            all_pos = np.array([self.label_data[s + f'_{freq}'] for s in self.signal_to_test
                                for freq in self.signal_to_test[s]])
            print(all_pos)

            r = self.run_parallel_back_test(
                df, signal_cols=all_pos, entry_price=self.__entry_pip / cst.FX_PIP[fx_pair],
                stop_loss=self.__stop_loss / cst.FX_PIP[fx_pair],
                take_profit=self.__take_profit / cst.FX_PIP[fx_pair],
                lot_size=self.__lot_size, valid=self.__valid,
            )
            r = pd.DataFrame(r, ).rename(columns=RESULT_COLS)
            print(r)
            print(self.label_data_r)
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

            return r
            results.append(r)

