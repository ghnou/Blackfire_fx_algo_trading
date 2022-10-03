import fx_trading.utils.constants as cst
from fx_trading.utils.data_interact import read_data
import pandas as pd
import numpy as np
from numba import njit, prange
import talib

import backtrader as bt
import time

TICK_1MIN = 60
TICK_5MIN = 60 * 5
TICK_15MIN = 60 * 15
TICK_60MIN = 60 * 60
TICK_4H = 60 * 60 * 4
TICK_1D = 60 * 60 * 24

PENDING_TRADE = 0
ACTIVE_TRADE = 1
CLOSED_TRADE = 2
INACTIVE_TRADE = -1
ORDER_BUY = 1
ORDER_SELL = -1


def add_candlestick(df):
    pass


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
    TOHLC = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE']

    MAPPING = {'15MIN': [2, 13, 35, 14, 15, 16, 17, 41]}

    @staticmethod
    @njit()
    def re_sample_data(df, n_candles=6):

        resample = np.empty((df.shape[0], 39 + n_candles), dtype=np.float64)
        resample[:] = np.NaN
        resample[:, :df.shape[1]] = df
        resample[:, 33:] = 0
        ohlc = np.empty((6, 4), dtype=np.float64)

        for i in prange(0, resample.shape[0]):

            resample[i, 2] = i + 1
            resample[i, 3] = (resample[i, 0] - 1) % 86400 // TICK_1MIN
            resample[i, 8] = (resample[i, 0] - 1) % 86400 // TICK_5MIN
            resample[i, 13] = (resample[i, 0] - 1) % 86400 // TICK_15MIN
            resample[i, 18] = (resample[i, 0] - 1) % 86400 // TICK_60MIN
            resample[i, 23] = (resample[i, 0] - 1) % 86400 // TICK_4H
            resample[i, 28] = (resample[i, 0] - 1) // TICK_1D

            # Initialisation
            if i == 0:
                resample[i, 4:8] = resample[i, 1]
                resample[i, 9:13] = resample[i, 1]
                resample[i, 14:18] = resample[i, 1]
                resample[i, 19:23] = resample[i, 1]
                resample[i, 24:28] = resample[i, 1]
                resample[i, 29:33] = resample[i, 1]
                ohlc[:, :] = resample[i, 1]
            else:
                for j in prange(6):
                    if resample[i, 5 * j + 3] != resample[i - 1, 5 * j + 3]:
                        ohlc[j, 0] = resample[i, 1]
                        ohlc[j, 1] = resample[i, 1]
                        ohlc[j, 2] = resample[i, 1]
                        ohlc[j, 3] = resample[i, 1]
                        resample[i, 33 + j] = 1
                        resample[i - 1, 33 + j] = -1

                ohlc[ohlc[:, 1] < resample[i, 1], 1] = resample[i, 1]
                ohlc[ohlc[:, 2] > resample[i, 1], 2] = resample[i, 1]
                ohlc[:, 3] = resample[i, 1]
                resample[i, 4:8] = ohlc[0]  # 1 MIN OHLC
                resample[i, 9:13] = ohlc[1]  # 5 MIN OHLC
                resample[i, 14:18] = ohlc[2]  # 5 MIN OHLC
                resample[i, 19:23] = ohlc[3]  # 5 MIN OHLC
                resample[i, 24:28] = ohlc[4]  # 5 MIN OHLC
                resample[i, 29:33] = ohlc[5]  # 5 MIN OHLC

        return resample

    def __init__(self, fx_data_dict):

        self.fx_manager = {}
        for fx_pair in fx_data_dict:
            self.fx_manager[fx_pair] = self.re_sample_data(fx_data_dict[fx_pair])
            self.add_candlestick_pattern(fx_pair, '15MIN')

    def add_candlestick_pattern(self, fx_pair, freq):
        """
        Description:
        ------------

        Add Candle stick pattern to the price.
        """
        df = self.fx_manager[fx_pair][:, self.MAPPING[freq]]

        df = df[df[:, 2] == -1]
        candle = getattr(talib, self.params['ind'])(df[:, 3], df[:, 4], df[:, 5], df[:, 6])
        self.fx_manager[fx_pair][df[:, 0].astype(np.int64), self.MAPPING[freq][-1]] = candle

        return df

    @staticmethod
    @njit()
    def run_backtest(df, entry_price, stop_loss, take_profit, lot_size=200_000):

        orders = np.empty((df[df[:, 41] != 0].shape[0], 13))
        order_number = 1
        for i in np.arange(df.shape[0]):
            signal = df[i, 41]
            close_price = df[i, 1]

            en = orders[:, 4]
            sl = orders[:, 5]
            tp = orders[:, 6]

            # Close Close Take Profit Buy Trade
            mask = (orders[:, 7] == ACTIVE_TRADE) & (orders[:, 1] == ORDER_BUY) & ((close_price >= tp) | (close_price <= sl))
            orders[mask, 3] = df[i, 0]
            orders[mask, 7] = CLOSED_TRADE
            orders[mask, 9] = df[i, 0]
            orders[mask, 11] = close_price
            orders[mask, 12] = (close_price - en[mask]) * lot_size

            # Close Close Take Profit Sell Trade
            mask = (orders[:, 7] == ACTIVE_TRADE) & (orders[:, 1] == ORDER_SELL) & ((close_price <= tp) | (close_price >= sl))
            orders[mask, 3] = df[i, 0]
            orders[mask, 7] = CLOSED_TRADE
            orders[mask, 9] = df[i, 0]
            orders[mask, 11] = close_price
            orders[mask, 12] = (en[mask] - close_price) * lot_size

            # Buy Order Execute
            mask = (orders[:, 7] == PENDING_TRADE) & (orders[:, 1] == ORDER_BUY) & (en >= close_price)
            orders[mask, 8] = df[i, 0]
            orders[mask, 10] = close_price
            orders[mask, 7] = ACTIVE_TRADE

            # SELL Order Execute
            mask = (orders[:, 7] == PENDING_TRADE) & (orders[:, 1] == ORDER_SELL) & (en <= close_price)
            orders[mask, 8] = df[i, 0]
            orders[mask, 10] = close_price
            orders[mask, 7] = ACTIVE_TRADE

            # Create New Orders
            if signal > 0:
                orders[order_number - 1, 0] = df[i, 0]       # Signal Time
                orders[order_number - 1, 1] = ORDER_BUY              # Order Type
                orders[order_number - 1, 2] = (df[i - 1, 15] - df[i - 1, 16]) * 10_000             # Order Type
                orders[order_number - 1, 3] = close_price    # Close Price
                orders[order_number - 1, 4] = close_price - entry_price   # Buy Price
                orders[order_number - 1, 5] = close_price - entry_price - stop_loss   # Stop Loss
                orders[order_number - 1, 6] = close_price - entry_price + take_profit   # Take Profit
                orders[order_number - 1, 7] = PENDING_TRADE   # Trade Status

                order_number += 1
            elif signal < 0:
                orders[order_number - 1, 0] = df[i, 0]  # Signal Time
                orders[order_number - 1, 1] = ORDER_SELL  # Order Type
                orders[order_number - 1, 2] = (df[i - 1, 15] - df[i - 1, 16]) * 10_000             # Order Type
                orders[order_number - 1, 3] = close_price  # Close Price
                orders[order_number - 1, 4] = close_price + entry_price  # Sell Price
                orders[order_number - 1, 5] = close_price + entry_price + stop_loss  # Stop Loss
                orders[order_number - 1, 6] = close_price + entry_price - take_profit  # Take Profit
                orders[order_number - 1, 7] = PENDING_TRADE  # Trade status

                order_number += 1

        return orders

    def run(self):

        for fx_pair in self.fx_manager:
            df = self.fx_manager[fx_pair]
            return self.run_backtest(df, 0, 45/10_000, 5 / 10_000)
        pass
data = pd.read_parquet('sample.parquet')
data['TIMESTAMP'] = data.index.astype(np.int64) // 10 ** 9
cols = ['TIMESTAMP', 'BID']
t = time.time()
bt = FastHighProbCandleStick(fx_data_dict={'eurgbp': data[cols].values})
df = bt.run()
print(time.time() - t)

cols = ['TIMESTAMP', 'ORDER_TYPE', 'SPREAD', 'PRICE', 'ENTRY', 'SL', 'TP', 'TRADE_STATUS', 'DT_OPEN',
        'DT_CLOSE', 'EXEC_ENTRY', 'EXEC_EXIT', 'PnL']
df = pd.DataFrame(df, columns=cols)
df['TIMESTAMP'] = df['TIMESTAMP'].astype(int)
df['DT_OPEN'] = df['DT_OPEN'].astype(int)
print(df)
#
# freq = ['1MIN', '5MIN', '15MIN', '1H', '4H', '1D']
# tohlc = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
# cols = ['TIMESTAMP', 'BID', 'TICK'] + [i + '_{}'.format(j) for j in freq for i in tohlc]
# cols += ['NEW_BAR_{}'.format(i) for i in freq] + ['ENGULFING']
# df = pd.DataFrame(df, columns=cols)
# df['TIMESTAMP'] = df['TIMESTAMP'].astype(int)
# df['ENGULFING'] = df['ENGULFING'].astype(int)
# df['TICK'] = df['TICK'].astype(int)
# df[['TIME_{}'.format(i) for i in freq]] = df[['TIME_{}'.format(i) for i in freq]].astype(int)
# print(df)
# df.to_csv('check.csv')

